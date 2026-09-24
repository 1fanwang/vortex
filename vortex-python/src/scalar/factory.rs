// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyBool;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyFloat;
use pyo3::types::PyInt;
use pyo3::types::PyList;
use pyo3::types::PyString;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability;
use vortex::dtype::StructFields;
use vortex::extension::datetime::Time;
use vortex::extension::datetime::TimeUnit;
use vortex::scalar::DecimalValue;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex_arrow::ArrowSessionExt;
use vortex_spatial::extension::native_geometry_scalar_from_wkb;

use crate::dtype::PyDType;
use crate::error::PyVortexResult;
use crate::scalar::PyScalar;
use crate::scalar::bool;
use crate::session::session;

#[pyfunction(name = "scalar")]
#[pyo3(signature = (value, *, dtype=None))]
pub fn scalar<'py>(
    py: Python<'py>,
    value: Bound<'py, PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<Bound<'py, PyScalar>> {
    PyScalar::init(
        py,
        scalar_helper(&value, dtype.as_ref().map(|dtype| dtype.inner()))?,
    )
}

pub fn scalar_helper(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyVortexResult<Scalar> {
    let scalar = scalar_helper_inner(value, dtype)?;

    // If a dtype was provided, attempt to  cast the scalar to that dtype.
    // This is a trivially cheap no-op if the scalar is already of the correct type.
    if let Some(dtype) = dtype {
        Ok(scalar.cast(dtype)?)
    } else {
        Ok(scalar)
    }
}

/// Attempts to convert the python object to a scalar, with a hint of the expected
/// dtype. It can assume that the scalar_helper function will perform a final cast to the correct
/// dtype if necessary.
fn scalar_helper_inner(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyResult<Scalar> {
    // If it's already a scalar, return it
    if let Ok(value) = value.cast::<PyScalar>() {
        return Ok(value.get().inner().clone());
    }

    // Otherwise, we start checking the known Python types.

    // None
    if value.is_none() {
        return Ok(Scalar::null(dtype.cloned().unwrap_or(DType::Null)));
    }

    // bool
    if let Ok(bool) = value.cast::<PyBool>() {
        return Ok(Scalar::bool(
            bool.extract::<bool>()?,
            Nullability::NonNullable,
        ));
    }

    // decimal
    if let Some(decimal_dtype) = dtype.and_then(|d| d.as_decimal_opt()) {
        let value = if let Ok(v) = value.extract::<i8>() {
            DecimalValue::I8(v)
        } else if let Ok(v) = value.extract::<i16>() {
            DecimalValue::I16(v)
        } else if let Ok(v) = value.extract::<i32>() {
            DecimalValue::I32(v)
        } else if let Ok(v) = value.extract::<i64>() {
            DecimalValue::I64(v)
        } else if let Ok(v) = value.extract::<i128>() {
            DecimalValue::I128(v)
        } else {
            return Err(PyValueError::new_err(
                "Value can't be represented as decimal",
            ));
        };
        return Ok(Scalar::decimal(
            value,
            *decimal_dtype,
            Nullability::NonNullable,
        ));
    }

    if let Ok(integer) = value.cast::<PyInt>() {
        return Ok(Scalar::primitive(
            integer.extract::<i64>()?,
            Nullability::NonNullable,
        ));
    }

    // float
    if let Ok(float) = value.cast::<PyFloat>() {
        return Ok(Scalar::primitive(
            float.extract::<f64>()?,
            Nullability::NonNullable,
        ));
    }

    // str
    if let Ok(string) = value.cast::<PyString>() {
        return Ok(Scalar::utf8(
            string.extract::<String>()?,
            Nullability::NonNullable,
        ));
    }

    // bytes
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(Scalar::binary(
            bytes.extract::<Vec<u8>>()?,
            Nullability::NonNullable,
        ));
    }

    // dict
    if let Ok(dict) = value.cast::<PyDict>() {
        // Extract the field names from the dictionary keys
        let names: FieldNames = dict
            .keys()
            .iter()
            .map(|key| key.extract::<String>())
            .map_ok(FieldName::from)
            .collect::<PyResult<Vec<FieldName>>>()?
            .into();

        if let Some(DType::Struct(dtype, nullability)) = dtype {
            if names != dtype.names() {
                return Err(PyValueError::new_err(format!(
                    "Dictionary field names {:?} do not match target dtype names {:?}",
                    names,
                    dtype.names()
                )));
            }

            let children: Vec<Scalar> = dict
                .values()
                .into_iter()
                .map(|item| scalar_helper_inner(&item, None))
                .try_collect()?;
            return Ok(Scalar::struct_(
                DType::Struct(dtype.clone(), *nullability),
                children,
            ));
        } else {
            let values: Vec<Scalar> = dict
                .values()
                .into_iter()
                .map(|value| scalar_helper_inner(&value, None))
                .try_collect()?;
            let dtype = DType::Struct(
                StructFields::new(
                    names,
                    values.iter().map(|value| value.dtype().clone()).collect(),
                ),
                Nullability::NonNullable,
            );
            return Ok(Scalar::struct_(dtype, values));
        };
    }

    if let Ok(list) = value.cast::<PyList>() {
        if let Some(DType::List(element_dtype, ..)) = dtype {
            let elements = list
                .iter()
                .map(|e| scalar_helper_inner(&e, Some(element_dtype)))
                .try_collect()?;
            Scalar::list(
                Arc::clone(element_dtype),
                elements,
                Nullability::NonNullable,
            );
        } else {
            // If no dtype was provided, we need to infer the element dtype from the list contents.
            // We do this in a greedy way taking the first element dtype we find.
            let mut elements = Vec::with_capacity(list.len());
            let mut element_dtype = None;

            for element in list.iter() {
                let scalar = scalar_helper_inner(&element, element_dtype.as_ref())?;
                if element_dtype.is_none() {
                    element_dtype = Some(scalar.dtype().clone());
                }
                elements.push(scalar);
            }

            return Ok(Scalar::list(
                element_dtype
                    .map(Arc::new)
                    // Empty list defaults to Null dtype
                    .unwrap_or_else(|| Arc::new(DType::Null)),
                elements,
                Nullability::NonNullable,
            ));
        }
    }

    // datetime.time
    let time_type = value
        .py()
        .import(intern!(value.py(), "datetime"))?
        .getattr(intern!(value.py(), "time"))?;
    if value.is_instance(&time_type)? {
        return Ok(time_scalar(value, dtype)?);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Cannot convert Python object to Vortex scalar: {}",
        value.get_type()
    )))
}

/// Convert a naive `datetime.time` into a Vortex `Time` scalar.
///
/// The unit is taken from `dtype` when it is a `Time` dtype, and otherwise defaults to
/// microseconds, the resolution of `datetime.time`. Converting to a coarser unit that would drop
/// a non-zero fraction of a second is an error rather than a silent truncation.
fn time_scalar(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyVortexResult<Scalar> {
    let py = value.py();
    if !value.getattr(intern!(py, "tzinfo"))?.is_none() {
        return Err(PyValueError::new_err(
            "Timezone-aware datetime.time values cannot be converted to a Vortex time scalar",
        )
        .into());
    }
    let hour: i64 = value.getattr(intern!(py, "hour"))?.extract()?;
    let minute: i64 = value.getattr(intern!(py, "minute"))?.extract()?;
    let second: i64 = value.getattr(intern!(py, "second"))?.extract()?;
    let microsecond: i64 = value.getattr(intern!(py, "microsecond"))?.extract()?;
    let micros = ((hour * 60 + minute) * 60 + second) * 1_000_000 + microsecond;

    let unit = dtype
        .and_then(DType::as_extension_opt)
        .and_then(|ext| ext.metadata_opt::<Time>())
        .copied()
        .unwrap_or(TimeUnit::Microseconds);

    let coarse = |per_unit: i64| -> PyResult<ScalarValue> {
        if micros % per_unit != 0 {
            return Err(PyValueError::new_err(format!(
                "Time value {micros}us cannot be represented in {unit} without losing precision"
            )));
        }
        let value = i32::try_from(micros / per_unit)
            .map_err(|_| PyValueError::new_err(format!("Time value does not fit in i32 {unit}")))?;
        Ok(ScalarValue::from(value))
    };
    let storage = match unit {
        TimeUnit::Nanoseconds => ScalarValue::from(micros * 1_000),
        TimeUnit::Microseconds => ScalarValue::from(micros),
        TimeUnit::Milliseconds => coarse(1_000)?,
        TimeUnit::Seconds => coarse(1_000_000)?,
        TimeUnit::Days => {
            return Err(PyValueError::new_err("Time type does not support time unit days").into());
        }
    };

    let ext = Time::try_new(unit, Nullability::NonNullable)?;
    Ok(Scalar::try_new(
        DType::Extension(ext.erased()),
        Some(storage),
    )?)
}

/// Construct a geometry scalar from its OGC Well-Known Binary (WKB) encoding.
///
/// The value is decoded into the native Vortex geometry type matching its kind (``Point``,
/// ``LineString``, ``Polygon``, ``MultiPoint``, ``MultiLineString`` or ``MultiPolygon``, in XY
/// with no coordinate reference system), which is the form Vortex's spatial functions and
/// pruning operate on. The resulting scalar can be used anywhere an expression is expected.
///
/// Parameters
/// ----------
/// wkb : :class:`bytes`
///     The WKB-encoded geometry, for example ``shapely.Point(1, 2).wkb``.
///
/// Returns
/// -------
/// :class:`vortex.Scalar`
///
/// Raises
/// ------
/// ValueError
///     If the bytes are not valid WKB, or encode an unsupported geometry such as a
///     ``GeometryCollection``.
///
/// Examples
/// --------
///
/// ```python
/// >>> import struct
/// >>> import vortex as vx
/// >>> point = vx.geometry_scalar(struct.pack("<BIdd", 1, 1, 1.0, 2.0))
/// >>> isinstance(point, vx.ExtensionScalar)
/// True
/// ```
#[pyfunction(name = "geometry_scalar")]
pub fn geometry_scalar<'py>(
    py: Python<'py>,
    wkb: &Bound<'py, PyBytes>,
) -> PyResult<Bound<'py, PyScalar>> {
    let scalar = native_geometry_scalar_from_wkb(wkb.as_bytes(), &session().arrow())
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .ok_or_else(|| {
            PyValueError::new_err("Unsupported WKB geometry type for a Vortex geometry scalar")
        })?;
    PyScalar::init(py, scalar)
}
