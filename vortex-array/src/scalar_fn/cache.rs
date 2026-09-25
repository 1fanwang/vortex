// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::OnceLock;

use vortex_error::VortexResult;

type Entry = Arc<dyn Any + Send + Sync>;

/// State a scalar function derives once for one node of a
/// [`BoundExpression`](crate::expr::BoundExpression) and reuses for every batch that node
/// evaluates, such as the set `list_contains` prepares from its constant list.
///
/// A bound node owns one cache and hands it to every [`ScalarFnArray`](crate::arrays::ScalarFnArray)
/// it is applied as; the function reads it through [`ExecutionArgs::cache`](super::ExecutionArgs).
/// The state may depend only on the node's function, its argument dtypes, and the values of its
/// constant arguments. A rewrite that keeps all three — slicing or filtering the node, or splitting
/// it over the chunks of an argument — carries the cache along; one that changes an argument's
/// dtype must not. A function should still check, as far as it cheaply can, that it is executing
/// with the constants and dtypes it derived state from.
///
/// Clones share the state. The cache takes no part in comparing or hashing the expression that
/// owns it.
#[derive(Clone, Default)]
pub struct ScalarFnCache(Arc<OnceLock<Entry>>);

impl ScalarFnCache {
    /// The cached value, computed by `init` on first use.
    ///
    /// Concurrent first uses may each run `init`; one result is kept. A value of another type in
    /// the cache is left in place, and `init`'s result is returned without being cached.
    pub fn get_or_try_init<T: Any + Send + Sync>(
        &self,
        init: impl FnOnce() -> VortexResult<T>,
    ) -> VortexResult<Arc<T>> {
        if let Some(cached) = self.get::<T>() {
            return Ok(cached);
        }
        if self.0.get().is_some() {
            return init().map(Arc::new);
        }
        let computed = Arc::new(init()?);
        let entry = self.0.get_or_init(|| Arc::clone(&computed) as Entry);
        Ok(Arc::clone(entry).downcast::<T>().unwrap_or(computed))
    }

    /// The cached value, if it has been computed and has type `T`.
    pub fn get<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        Arc::clone(self.0.get()?).downcast::<T>().ok()
    }
}

impl Debug for ScalarFnCache {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScalarFnCache")
            .field("initialized", &self.0.get().is_some())
            .finish()
    }
}

/// Every cache is equal: it holds derived state, not part of the node's meaning.
impl PartialEq for ScalarFnCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for ScalarFnCache {}

impl Hash for ScalarFnCache {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}
