// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::scalar_fn::fns::list_contains::ListContainsElementKernel;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::list_contains::ListContainsSet;

/// Membership of chunked needles in a constant set, which is prepared once and probes every chunk.
///
/// The chunks are probed as the set's canonical implementation probes them, so a chunk's own
/// encoding-specific `list_contains` is not consulted: preparing the set per chunk is what this
/// avoids.
impl ListContainsElementKernel for Chunked {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(set) = ListContainsSet::try_new(list, element.dtype(), options, ctx)? else {
            return Ok(None);
        };
        let set = set.prepare(ctx)?;
        let chunks = element
            .iter_chunks()
            .map(|chunk| set.contains(chunk, ctx))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Some(
            ChunkedArray::try_new(chunks, set.dtype())?.into_array(),
        ))
    }
}
