// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::black_box;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::LazyLock;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

const ARRAY_SIZE: usize = 100_000;
const NUM_ACCESSES: usize = 100;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn array() -> ArrayRef {
    let lhs =
        PrimitiveArray::from_option_iter((0..ARRAY_SIZE).map(|i| (i % 7 != 0).then_some(i as i64)))
            .into_array();
    let rhs = PrimitiveArray::from_iter((0..ARRAY_SIZE).map(|i| i as i64)).into_array();
    let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Add).erased();
    ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])
        .unwrap()
        .into_array()
}

fn indices() -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..ARRAY_SIZE))
        .collect()
}

#[divan::bench]
fn probe_scalar_fn_once(bencher: Bencher) {
    let array = array();
    let indices = indices();

    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            for &index in indices.iter() {
                black_box(array.execute_scalar(index, ctx).unwrap());
            }
        });
}

#[divan::bench]
fn probe_scalar_fn_repeated(bencher: Bencher) {
    let array = array();
    let indices = indices();

    bencher
        .with_inputs(|| {
            (
                array.repeated_probe(),
                &indices,
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(probe, indices, ctx)| {
            for &index in indices.iter() {
                black_box(probe.execute_scalar(index, ctx).unwrap());
            }
        });
}
