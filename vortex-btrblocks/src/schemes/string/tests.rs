// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Masked;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::display::DisplayOptions;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[test]
fn test_strings() -> VortexResult<()> {
    let mut strings = Vec::new();
    for _ in 0..1024 {
        strings.push(Some("hello-world-1234"));
    }
    for _ in 0..1024 {
        strings.push(Some("hello-world-56789"));
    }
    let strings = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));

    let array_ref = strings.into_array();
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert_eq!(compressed.len(), 2048);

    let display = compressed
        .display_as(DisplayOptions::MetadataOnly)
        .to_string()
        .to_lowercase();
    assert_eq!(display, "vortex.dict(utf8, len=2048)");

    Ok(())
}

/// Mostly-null strings whose valid values are all equal compress to a masked constant.
#[test]
fn test_mostly_null_constant() -> VortexResult<()> {
    let mut strings = VarBinViewBuilder::with_capacity_in(
        DType::Utf8(Nullability::Nullable),
        100,
        vortex_buffer::BufferAllocatorRef::statically_allocated(),
    );
    strings.append_nulls(99);

    strings.append_value("one little string");

    let strings = strings.finish_into_varbinview();

    let array_ref = strings.into_array();
    let btr = BtrBlocksCompressor::default();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = btr.compress(&array_ref, &mut ctx)?;

    assert!(compressed.is::<Masked>());
    assert!(compressed.children()[0].is::<Constant>());
    assert_arrays_eq!(compressed, array_ref, &mut ctx);

    Ok(())
}
