// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

pub use kernel::*;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyColumnar;
use crate::ArrayRef;
use crate::CanonicalView;
use crate::ColumnarView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Bool;
use crate::arrays::Decimal;
use crate::arrays::Primitive;
use crate::arrays::ScalarFnArray;
use crate::builders::builder_with_capacity_in;
use crate::dtype::DType;
use crate::expr::BoundExpression;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::is_not_null_node;

/// An expression that replaces null values in the input with a fill value.
#[derive(Clone)]
pub struct FillNull;

impl FillNull {
    /// Creates a lazy operation that replaces null input values with `fill_value`.
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths or incompatible dtypes.
    pub fn try_new(input: ArrayRef, fill_value: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(FillNull.bind(EmptyOptions), [input, fill_value])
    }

    /// Returns whether fill-null can execute values of `dtype`.
    pub(crate) fn supports_dtype(dtype: &DType) -> bool {
        !matches!(dtype, DType::Null) && has_builder(dtype)
    }
}

impl ScalarFnVTable for FillNull {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.fill_null");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            1 => ChildName::from("fill_value"),
            _ => unreachable!("Invalid child index {} for FillNull expression", child_idx),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            arg_dtypes[0].eq_ignore_nullability(&arg_dtypes[1]),
            "fill_null requires input and fill value to have the same base type, got {} and {}",
            arg_dtypes[0],
            arg_dtypes[1]
        );
        // The result dtype takes the nullability of the fill value.
        Ok(arg_dtypes[0]
            .clone()
            .with_nullability(arg_dtypes[1].nullability()))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let fill_value = args.get(1)?;

        let fill_scalar = fill_value
            .as_constant()
            .ok_or_else(|| vortex_err!("fill_null fill_value must be a constant/scalar"))?;

        vortex_ensure!(
            !fill_scalar.is_null(),
            "fill_null requires a non-null fill value"
        );

        let Some(columnar) = input.as_opt::<AnyColumnar>() else {
            let input = input.execute::<ArrayRef>(ctx)?;
            return Ok(FillNull::try_new(input, fill_value)?.into_array());
        };

        match columnar {
            ColumnarView::Canonical(canonical) => fill_null_canonical(canonical, &fill_scalar, ctx),
            ColumnarView::Constant(constant) => fill_null_constant(constant, &fill_scalar),
        }
    }

    fn simplify(
        &self,
        _options: &Self::Options,
        expr: &BoundExpression,
    ) -> VortexResult<Option<BoundExpression>> {
        if !expr.child(0).dtype().is_nullable() {
            return Ok(Some(expr.child(0).clone()));
        }

        Ok(None)
    }

    fn validity<T: ReduceNode>(
        &self,
        _options: &Self::Options,
        node: &T,
    ) -> VortexResult<ReduceNodeValidity<T>> {
        // After fill_null, the result validity depends on the fill value's nullability.
        // If fill_value is non-nullable, the result is always valid.
        Ok(ReduceNodeValidity::Reduced(is_not_null_node(
            &node.child(1),
        )?))
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        // This function replaces null input values instead of propagating them.
        false
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Fill nulls on a canonical array by directly dispatching to the appropriate kernel.
///
/// Returns the filled array, or bails if no kernel is registered for the canonical type.
fn fill_null_canonical(
    canonical: CanonicalView<'_>,
    fill_value: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let arr = canonical.to_array_ref();
    if let Some(result) = short_circuit(&arr, fill_value)? {
        return Ok(result);
    }
    match canonical {
        CanonicalView::Bool(a) => <Bool as FillNullKernel>::fill_null(a, fill_value, ctx)?
            .ok_or_else(|| vortex_err!("FillNullKernel for BoolArray returned None")),
        CanonicalView::Primitive(a) => {
            <Primitive as FillNullKernel>::fill_null(a, fill_value, ctx)?
                .ok_or_else(|| vortex_err!("FillNullKernel for PrimitiveArray returned None"))
        }
        CanonicalView::Decimal(a) => <Decimal as FillNullKernel>::fill_null(a, fill_value, ctx)?
            .ok_or_else(|| vortex_err!("FillNullKernel for DecimalArray returned None")),
        _ if FillNull::supports_dtype(arr.dtype()) => fill_null_with_builder(&arr, fill_value, ctx),
        other => vortex_bail!(
            "No FillNullKernel for canonical array {}",
            other.to_array_ref().encoding_id()
        ),
    }
}

fn has_builder(dtype: &DType) -> bool {
    match dtype {
        DType::Null
        | DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => true,
        DType::List(element, _) | DType::FixedSizeList(element, ..) => has_builder(element),
        DType::Map(map, _) => has_builder(&map.key_dtype()) && has_builder(&map.value_dtype()),
        DType::Struct(fields, _) => fields.fields().all(|field| has_builder(&field)),
        DType::Union(..) | DType::Variant(_) => false,
        DType::Extension(ext) => has_builder(ext.storage_dtype()),
    }
}

fn fill_null_with_builder(
    array: &ArrayRef,
    fill_value: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut builder = builder_with_capacity_in(fill_value.dtype(), array.len(), ctx.allocator());
    let mut probe = array.repeated_probe();
    for index in 0..array.len() {
        let value = probe.execute_scalar(index, ctx)?;
        if value.is_null() {
            builder.append_scalar(fill_value)?;
        } else {
            builder.append_scalar(&value.cast(fill_value.dtype())?)?;
        }
    }
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use super::*;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity_in;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::MapDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::dtype::UnionVariants;
    use crate::expr::fill_null;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::scalar::Scalar;

    #[test]
    fn dtype() {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        assert_eq!(
            fill_null(root(), lit(0i32)).return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    #[test]
    fn replace_children() {
        let expr = fill_null(root(), lit(0i32));
        expr.with_children(vec![root(), lit(0i32)])
            .vortex_expect("operation should succeed in test");
    }

    #[test]
    fn evaluate() {
        let mut ctx = array_session().create_execution_ctx();
        let test_array =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)])
                .into_array();

        let expr = fill_null(root(), lit(42i32));
        let result = test_array.apply(&expr).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([1i32, 42, 3, 42, 5]),
            &mut ctx
        );
    }

    #[test]
    fn evaluate_struct_field() {
        let mut ctx = array_session().create_execution_ctx();
        let test_array = StructArray::from_fields(&[(
            "a",
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
        )])
        .unwrap()
        .into_array();

        let expr = fill_null(get_item("a", root()), lit(0i32));
        let result = test_array.apply(&expr).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 0, 3]), &mut ctx);
    }

    #[test]
    fn evaluate_non_nullable_input() {
        let mut ctx = array_session().create_execution_ctx();
        let test_array = buffer![1i32, 2, 3].into_array();
        let expr = fill_null(root(), lit(0i32));
        let result = test_array.apply(&expr).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 2, 3]), &mut ctx);
    }

    #[test]
    fn evaluate_builder_backed_dtypes() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        for (name, value, fill) in supported_scalar_pairs()? {
            let mut builder = builder_with_capacity_in(value.dtype(), 2, ctx.allocator());
            builder.append_scalar(&value)?;
            builder.append_scalar(&Scalar::null(value.dtype().clone()))?;
            let input = builder.finish();

            let result = input
                .fill_null(fill.clone())?
                .execute::<Canonical>(&mut ctx)?
                .into_array();
            assert_eq!(result.dtype(), fill.dtype(), "{name}");
            assert_eq!(
                result.execute_scalar(0, &mut ctx)?,
                value.cast(fill.dtype())?,
                "{name}"
            );
            assert_eq!(result.execute_scalar(1, &mut ctx)?, fill, "{name}");
        }
        Ok(())
    }

    #[test]
    fn supports_builder_backed_dtypes_only() -> VortexResult<()> {
        for (name, value, _) in supported_scalar_pairs()? {
            assert!(FillNull::supports_dtype(value.dtype()), "{name}");
        }

        let union = UnionVariants::try_new(
            ["number"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            vec![0],
        )?;
        for dtype in [
            DType::Null,
            DType::Union(union, Nullability::Nullable),
            DType::Variant(Nullability::Nullable),
            DType::List(
                Arc::new(DType::Variant(Nullability::Nullable)),
                Nullability::Nullable,
            ),
        ] {
            assert!(!FillNull::supports_dtype(&dtype), "{dtype}");
        }
        Ok(())
    }

    fn supported_scalar_pairs() -> VortexResult<Vec<(&'static str, Scalar, Scalar)>> {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let map_dtype = MapDType::try_new(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::NonNullable),
            false,
        )?;
        let struct_fields = StructFields::from_iter([(
            "value",
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )]);

        Ok(vec![
            (
                "utf8",
                Scalar::utf8("a", Nullability::Nullable),
                Scalar::utf8("fallback", Nullability::NonNullable),
            ),
            (
                "binary",
                Scalar::binary(vec![1], Nullability::Nullable),
                Scalar::binary(vec![2], Nullability::NonNullable),
            ),
            (
                "list",
                Scalar::list(
                    Arc::clone(&element_dtype),
                    vec![Scalar::from(1i32)],
                    Nullability::Nullable,
                ),
                Scalar::list(
                    Arc::clone(&element_dtype),
                    vec![Scalar::from(2i32)],
                    Nullability::NonNullable,
                ),
            ),
            (
                "fixed_size_list",
                Scalar::fixed_size_list(
                    Arc::clone(&element_dtype),
                    vec![Scalar::from(1i32), Scalar::from(2i32)],
                    Nullability::Nullable,
                ),
                Scalar::fixed_size_list(
                    Arc::clone(&element_dtype),
                    vec![Scalar::from(3i32), Scalar::from(4i32)],
                    Nullability::NonNullable,
                ),
            ),
            (
                "map",
                Scalar::map(
                    DType::Map(map_dtype.clone(), Nullability::Nullable),
                    [(
                        Scalar::from(1i32),
                        Scalar::utf8("a", Nullability::NonNullable),
                    )],
                ),
                Scalar::map(
                    DType::Map(map_dtype, Nullability::NonNullable),
                    [(
                        Scalar::from(2i32),
                        Scalar::utf8("b", Nullability::NonNullable),
                    )],
                ),
            ),
            (
                "struct",
                Scalar::struct_(
                    DType::Struct(struct_fields.clone(), Nullability::Nullable),
                    [Scalar::from(1i32)],
                ),
                Scalar::struct_(
                    DType::Struct(struct_fields, Nullability::NonNullable),
                    [Scalar::from(2i32)],
                ),
            ),
            (
                "extension",
                Scalar::extension::<Date>(TimeUnit::Days, Scalar::from(Some(1i32))),
                Scalar::extension::<Date>(TimeUnit::Days, Scalar::from(2i32)),
            ),
        ])
    }

    #[test]
    fn test_display() {
        let expr = fill_null(get_item("value", root()), lit(0i32));
        assert_eq!(expr.to_string(), "vortex.fill_null($.value, 0i32)");
    }
}
