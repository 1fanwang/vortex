// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Probing a constant set: the probe structure is built once from a [`ListContainsSet`], then run
//! over any number of needle arrays, such as the chunks of one column.

use std::hash::BuildHasher;

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashTable;
use vortex_utils::aliases::hash_map::HashTableEntry;
use vortex_utils::aliases::hash_map::RandomState;
use vortex_utils::iter::ReduceBalancedIterExt;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::varbinview::BinaryView;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::collect_bits;
use crate::scalar_fn::fns::list_contains::ListContainsSet;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

/// A set whose span of values needs at most this many bits per element is probed through a bitmap
/// over the span, bounding the bitmap to a few words per element.
const BITMAP_BITS_PER_ELEMENT: u128 = 64;
/// A span this narrow is probed through a bitmap whatever the size of the set.
const BITMAP_MIN_BITS: u128 = 1 << 12;

/// A [`ListContainsSet`] with the structure that probes it, prepared once and then run over any
/// number of needle arrays of the set's dtype.
///
/// Primitive and string needles take a single pass against a bitmap, a sorted slice or a hash set;
/// needles of any other dtype fall back to one equality comparison per element, OR-ed together.
pub struct PreparedSet {
    set: ListContainsSet,
    probe: Probe,
}

enum Probe {
    /// Integers, or floats by their bit patterns, spanning a dense range: one bit per value of the
    /// span above `min_offset`, the smallest element as a `usize`.
    Bitmap {
        min_offset: usize,
        span: usize,
        bitmap: BitBuffer,
    },
    /// Integers, or floats by their bit patterns, sorted without duplicates.
    Sorted(PrimitiveArray),
    /// UTF-8 or binary elements, found through a table of their indices hashed by their bytes, so
    /// that no element is copied.
    Bytes {
        elements: VarBinViewArray,
        hasher: RandomState,
        table: HashTable<u32>,
    },
    /// The elements of any other dtype, compared one at a time.
    Compare(Vec<Scalar>),
}

impl ListContainsSet {
    /// Builds the structure that probes this set.
    pub fn prepare(self, ctx: &mut ExecutionCtx) -> VortexResult<PreparedSet> {
        let probe = match self.elements().dtype() {
            DType::Primitive(ptype, _) => {
                let ptype = bit_pattern_ptype(*ptype);
                let elements = self
                    .elements()
                    .clone()
                    .execute::<PrimitiveArray>(ctx)?
                    .reinterpret_cast(ptype);
                match_each_integer_ptype!(ptype, |T| { integer_probe::<T>(elements) })
            }
            DType::Utf8(_) | DType::Binary(_) => {
                bytes_probe(self.elements().clone().execute::<VarBinViewArray>(ctx)?)
            }
            _ => Probe::Compare(
                (0..self.elements().len())
                    .map(|idx| self.elements().execute_scalar(idx, ctx))
                    .collect::<VortexResult<_>>()?,
            ),
        };
        Ok(PreparedSet { set: self, probe })
    }
}

impl PreparedSet {
    /// The dtype of every result, as the expression declares it.
    pub fn dtype(&self) -> DType {
        DType::Bool(self.set.nullability)
    }

    /// Whether each of `needles`, which have the set's dtype, is a member of the set.
    ///
    /// The result has the nullability the expression declares, whatever the needles' encoding.
    pub fn contains(&self, needles: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        match &self.probe {
            Probe::Bitmap { .. } | Probe::Sorted(_) => self.contains_primitive(needles, ctx),
            Probe::Bytes {
                elements,
                hasher,
                table,
            } => self.contains_bytes(elements, hasher, table, needles, ctx),
            Probe::Compare(elements) => self.contains_by_comparison(elements, needles, ctx),
        }
    }

    /// A float is a member exactly when the compare kernel would call it equal to an element, which
    /// is when their bit patterns match — distinguishing `-0.0` from `0.0` and one NaN payload from
    /// another — so floats are probed by their bits, as integers.
    fn contains_primitive(
        &self,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = needles.clone().execute::<PrimitiveArray>(ctx)?;
        let ptype = bit_pattern_ptype(primitive.ptype());
        let values = primitive.reinterpret_cast(ptype);
        let bits = match_each_integer_ptype!(ptype, |T| {
            self.probe
                .integer_bits(values.as_slice::<T>(), ctx.allocator())
        });
        self.set.finish(bits, primitive.validity()?)
    }

    fn contains_bytes(
        &self,
        elements: &VarBinViewArray,
        hasher: &RandomState,
        table: &HashTable<u32>,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let element_views = elements.views();
        let element_buffers = data_buffers(elements);
        let array = needles.clone().execute::<VarBinViewArray>(ctx)?;
        let buffers = data_buffers(&array);
        let bits = collect_bits(
            array.views(),
            |view: BinaryView| {
                let value = view_bytes(&view, &buffers);
                table
                    .find(hasher.hash_one(value), |&idx| {
                        view_bytes(&element_views[idx as usize], &element_buffers) == value
                    })
                    .is_some()
            },
            ctx.allocator(),
        );
        self.set.finish(bits, array.validity()?)
    }

    /// One equality per element, folded with Kleene OR.
    ///
    /// A null needle compares null to every element and so stays null. The elements hold no null,
    /// so under SQL null semantics one all-null term stands in for however many the list held:
    /// Kleene OR leaves a match `true` and turns a non-match into `null`.
    fn contains_by_comparison(
        &self,
        elements: &[Scalar],
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let len = needles.len();
        let unknown = self.set.non_match_is_unknown().then(|| {
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), len).into_array()
        });
        let result = elements
            .iter()
            .map(|element| {
                Ok(Binary::try_new(
                    ConstantArray::new(element.clone(), len).into_array(),
                    needles.clone(),
                    Operator::Eq,
                )?
                .into_array())
            })
            .collect::<VortexResult<Vec<_>>>()?
            .into_iter()
            .chain(unknown)
            .try_reduce_balanced(|acc, res| acc.binary(res, Operator::Or))?;

        let Some(result) = result else {
            // The list is empty or held nothing but nulls, and a non-match is known to be false:
            // nothing matches, and the set settles what a null needle answers.
            return self.set.finish(
                BitBuffer::full_in(false, len, ctx.allocator().clone()),
                needles.validity()?,
            );
        };
        // A comparison takes its nullability from the element's dtype, which need not be the
        // result's: off SQL null semantics a nullable element dtype decides nothing once the null
        // elements are gone.
        let dtype = self.dtype();
        if result.dtype() == &dtype {
            Ok(result)
        } else {
            result.cast(dtype)
        }
    }
}

impl Probe {
    /// One bit per needle, set when the needle is an element.
    fn integer_bits<T: IntegerPType>(
        &self,
        needles: &[T],
        allocator: &BufferAllocatorRef,
    ) -> BitBuffer {
        match self {
            Self::Bitmap {
                min_offset,
                span,
                bitmap,
            } => collect_bits(
                needles,
                // A needle below the smallest element wraps past `span`, so one comparison checks
                // both bounds.
                |needle| {
                    let offset = needle.as_().wrapping_sub(*min_offset);
                    offset <= *span && bitmap.value(offset)
                },
                allocator,
            ),
            Self::Sorted(sorted) => {
                let sorted = sorted.as_slice::<T>();
                collect_bits(
                    needles,
                    |needle| sorted.binary_search(&needle).is_ok(),
                    allocator,
                )
            }
            Self::Bytes { .. } | Self::Compare(_) => {
                unreachable!("integer needles meet an integer probe")
            }
        }
    }
}

/// A table of the elements' indices, hashed by their bytes, holding each distinct value once.
fn bytes_probe(elements: VarBinViewArray) -> Probe {
    let hasher = RandomState::default();
    let mut table = HashTable::with_capacity(elements.len());
    {
        let views = elements.views();
        let buffers = data_buffers(&elements);
        let bytes = |idx: u32| view_bytes(&views[idx as usize], &buffers);
        for (idx, view) in views.iter().enumerate() {
            let value = view_bytes(view, &buffers);
            if let HashTableEntry::Vacant(vacant) = table.entry(
                hasher.hash_one(value),
                |&other| bytes(other) == value,
                |&other| hasher.hash_one(bytes(other)),
            ) {
                vacant.insert(
                    u32::try_from(idx).vortex_expect("a list holds fewer than 2^32 elements"),
                );
            }
        }
    }
    Probe::Bytes {
        elements,
        hasher,
        table,
    }
}

/// The host slices of an array's data buffers, indexed by a view's buffer index.
fn data_buffers(array: &VarBinViewArray) -> Vec<&[u8]> {
    (0..array.data_buffers().len())
        .map(|idx| array.buffer(idx).as_slice())
        .collect()
}

/// The bytes a view points at, inlined in the view itself or out of line in one of `buffers`.
fn view_bytes<'a>(view: &'a BinaryView, buffers: &[&'a [u8]]) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let reference = view.as_view();
        &buffers[reference.buffer_index as usize][reference.as_range()]
    }
}

/// The integer type with a float's bit pattern, or the type itself for an integer.
fn bit_pattern_ptype(ptype: PType) -> PType {
    match ptype {
        PType::F16 => PType::U16,
        PType::F32 => PType::U32,
        PType::F64 => PType::U64,
        _ => ptype,
    }
}

/// A bitmap over the elements' span when the span is dense, and a sorted slice otherwise.
///
/// A hash set and, for a handful of elements, a linear scan both lost to the binary search at every
/// set size measured by the `list_contains_set` benchmark, up to 16 384 elements.
fn integer_probe<T: IntegerPType>(elements: PrimitiveArray) -> Probe {
    let values = elements.as_slice::<T>();
    let (Some(&min), Some(&max)) = (values.iter().min(), values.iter().max()) else {
        return Probe::Sorted(elements);
    };
    let span = integer_span(min, max);

    // The offset from `min` is computed in `usize`, which then has to hold every value of `T`.
    if size_of::<T>() <= size_of::<usize>()
        && span < (values.len() as u128 * BITMAP_BITS_PER_ELEMENT).max(BITMAP_MIN_BITS)
    {
        let span = usize::try_from(span).vortex_expect("span bounded by the bitmap limit");
        let mut bitmap = BitBufferMut::new_unset(span + 1);
        let min_offset: usize = min.as_();
        for &value in values {
            bitmap.set(value.as_().wrapping_sub(min_offset));
        }
        return Probe::Bitmap {
            min_offset,
            span,
            bitmap: bitmap.freeze(),
        };
    }

    // A set normalized while the expression was optimized arrives sorted already.
    if values.is_sorted_by(|a, b| a < b) {
        return Probe::Sorted(elements);
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    Probe::Sorted(PrimitiveArray::new(
        Buffer::from(sorted),
        Validity::NonNullable,
    ))
}

/// How far `max` lies above `min`, wider than either so that it cannot overflow.
fn integer_span<T: IntegerPType>(min: T, max: T) -> u128 {
    let wide = |value: T| {
        value
            .to_i128()
            .vortex_expect("an integer ptype fits in i128")
    };
    u128::try_from(wide(max) - wide(min)).vortex_expect("max is at least min")
}
