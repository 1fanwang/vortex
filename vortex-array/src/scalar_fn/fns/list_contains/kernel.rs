// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::BoolArray;
use crate::arrays::ListViewArray;
use crate::arrays::ScalarFn;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::list_contains::ListContains as ListContainsExpr;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::validity::Validity;

/// Check list-contains without reading buffers (metadata-only).
///
/// This trait dispatches on the **element** (needle) child at index 1 of the `ListContains`
/// expression. `Self` is the concrete element encoding, while the list (haystack) is passed as an
/// opaque `&ArrayRef`; [`ListContainsListReduce`] is the mirror image, dispatching on the list.
///
/// Return `None` if the operation cannot be resolved from metadata alone.
pub trait ListContainsElementReduce: VTable {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Check list-contains, potentially reading buffers.
///
/// Like [`ListContainsElementReduce`], this dispatches on the **element** (needle) child at
/// index 1. Unlike the reduce variant, implementations may read and execute on buffers via
/// the provided [`ExecutionCtx`].
pub trait ListContainsElementKernel: VTable {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Check list-contains without reading buffers (metadata-only).
///
/// This trait dispatches on the **list** (haystack) child at index 0 of the `ListContains`
/// expression, for encodings with a specialized list representation. `Self` is the concrete list
/// encoding, while the needle is passed as an opaque `&ArrayRef`.
///
/// Return `None` if the operation cannot be resolved from metadata alone.
pub trait ListContainsListReduce: VTable {
    fn list_contains(
        list: ArrayView<'_, Self>,
        needle: &ArrayRef,
        options: &ListContainsOptions,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Check list-contains, potentially reading buffers.
///
/// Like [`ListContainsListReduce`], this dispatches on the **list** (haystack) child at index 0.
/// Unlike the reduce variant, implementations may read and execute on buffers via the provided
/// [`ExecutionCtx`].
pub trait ListContainsListKernel: VTable {
    fn list_contains(
        list: ArrayView<'_, Self>,
        needle: &ArrayRef,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// The constant haystack of a `list_contains`, prepared for one pass over a column of needles.
///
/// This is the `IN` shape, `list_contains(lit([...]), column)`: the list's elements are executed
/// once into [`elements`](Self::elements), a kernel builds a probe structure from them and tests
/// every needle against it, then hands the membership bits back to [`finish`](Self::finish).
/// Shared by the canonical implementations and the element kernels, so that the result nullability
/// and the SQL null semantics have one definition.
///
/// A null element is dropped from the elements — it never equals a needle — and only decides
/// whether a non-match is [unknown](Self::non_match_is_unknown).
///
/// Preparing the set executes the list, so this serves [`ListContainsElementKernel`]. A
/// [`ListContainsElementReduce`] rule, which may not read buffers, works from the list's scalar.
pub struct ListContainsSet {
    elements: ArrayRef,
    nullability: Nullability,
    non_match_is_unknown: bool,
}

impl ListContainsSet {
    /// Prepares the constant list `list` to be probed by needles of dtype `needle_dtype`.
    ///
    /// Returns `None` when there is no set to probe: a list that is not a constant, is null, or is
    /// empty, or a needle whose dtype does not match the list's elements.
    pub fn try_new(
        list: &ArrayRef,
        needle_dtype: &DType,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self>> {
        let DType::List(element_dtype, _) = list.dtype() else {
            return Ok(None);
        };
        // A needle of a different dtype is an invalid expression, which the scalar function
        // reports as an error.
        if !element_dtype.eq_ignore_nullability(needle_dtype) {
            return Ok(None);
        }
        // Every needle is probed against the same set, so the haystack has to be one constant
        // list, and a null one holds no elements at all.
        let Some(list_scalar) = list.as_constant() else {
            return Ok(None);
        };
        if list.is_empty() || list_scalar.is_null() {
            return Ok(None);
        }

        // One row holds every element of a constant list, and executing it hands the elements over
        // as an array rather than as a scalar each.
        let list_view = list.slice(0..1)?.execute::<ListViewArray>(ctx)?;
        let offset = list_view.offset_at(0);
        let size = list_view.size_at(0);
        if size == 0 {
            return Ok(None);
        }
        let elements = list_view.elements().slice(offset..offset + size)?;

        let valid = elements.validity()?.execute_mask(size, ctx)?;
        let non_match_is_unknown = options.sql_null_semantics && !valid.all_true();
        // A null element never equals a needle, so a probe is better off without it.
        let elements = if valid.all_true() {
            elements
        } else {
            elements.filter(valid)?
        };

        Ok(Some(Self {
            elements,
            nullability: options.result_nullability(list.dtype(), needle_dtype),
            non_match_is_unknown,
        }))
    }

    /// The list's non-null elements, one row each, in the dtype of the needles.
    pub fn elements(&self) -> &ArrayRef {
        &self.elements
    }

    /// Whether a needle matching no element answers `null` rather than `false`.
    ///
    /// A null element under [`ListContainsOptions::sql_null_semantics`] makes the comparison
    /// unknown, the three-valued semantics of SQL `IN`.
    pub fn non_match_is_unknown(&self) -> bool {
        self.non_match_is_unknown
    }

    /// The nullability of the result, as declared by the expression's return dtype.
    pub fn nullability(&self) -> Nullability {
        self.nullability
    }

    /// Assembles the result from one membership bit per needle and the needles' validity.
    ///
    /// When a non-match is unknown, only the rows that matched stay valid.
    pub fn finish(&self, bits: BitBuffer, needle_validity: Validity) -> VortexResult<ArrayRef> {
        let validity = if self.non_match_is_unknown {
            needle_validity.and(Validity::from(bits.clone()))?
        } else {
            needle_validity
        };
        Ok(BoolArray::new(bits, validity.union_nullability(self.nullability)).into_array())
    }
}

/// Adaptor that wraps a [`ListContainsElementReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct ListContainsElementReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for ListContainsElementReduceAdaptor<V>
where
    V: ListContainsElementReduce,
{
    type Parent = ExactScalarFn<ListContainsExpr>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the element/needle child (index 1), not the list child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let list = scalar_fn_array.get_child(0);
        <V as ListContainsElementReduce>::list_contains(list, array, parent.options)
    }
}

/// Adaptor that wraps a [`ListContainsElementKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct ListContainsElementExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for ListContainsElementExecuteAdaptor<V>
where
    V: ListContainsElementKernel,
{
    type Parent = ExactScalarFn<ListContainsExpr>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the element/needle child (index 1), not the list child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let list = scalar_fn_array.get_child(0);
        <V as ListContainsElementKernel>::list_contains(list, array, parent.options, ctx)
    }
}

/// Adaptor that wraps a [`ListContainsListReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct ListContainsListReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for ListContainsListReduceAdaptor<V>
where
    V: ListContainsListReduce,
{
    type Parent = ExactScalarFn<ListContainsExpr>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the list child (index 0), not the element/needle child (index 1).
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let needle = scalar_fn_array.get_child(1);
        <V as ListContainsListReduce>::list_contains(array, needle, parent.options)
    }
}

/// Adaptor that wraps a [`ListContainsListKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct ListContainsListExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for ListContainsListExecuteAdaptor<V>
where
    V: ListContainsListKernel,
{
    type Parent = ExactScalarFn<ListContainsExpr>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the list child (index 0), not the element/needle child (index 1).
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let needle = scalar_fn_array.get_child(1);
        <V as ListContainsListKernel>::list_contains(array, needle, parent.options, ctx)
    }
}
