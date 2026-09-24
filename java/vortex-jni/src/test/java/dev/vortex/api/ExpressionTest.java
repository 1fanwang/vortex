// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.jni.NativeLoader;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public final class ExpressionTest {
    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    @Test
    public void rowIdxBuildsAndComposes() {
        assertNotNull(Expression.rowIdx());
        // Mirrors `gt(row_idx(), lit(...))` on the Rust side: the row-index expression
        // composes like any other.
        assertNotNull(Expression.binary(Expression.BinaryOp.LT, Expression.rowIdx(), Expression.literal(5L)));
    }

    @Test
    public void literalDecimalRejectsValuesLargerThan32Bytes() {
        BigInteger tooLarge = BigInteger.ONE.shiftLeft(256);
        assertEquals(33, tooLarge.toByteArray().length);

        RuntimeException exception =
                assertThrows(RuntimeException.class, () -> Expression.literalDecimal(tooLarge, 76, 0));
        assertTrue(exception.getMessage().contains("Decimal value must fit with 32 bytes"));
    }

    @Test
    public void packComposes() {
        assertNotNull(Expression.pack(
                new String[] {"x", "y", "z"},
                new Expression[] {Expression.column("a"), Expression.literal(5L), Expression.rowIdx()},
                true));
    }

    @Test
    public void mergeComposes() {
        // Default duplicate handling (ERROR).
        assertNotNull(Expression.merge(Expression.column("a"), Expression.column("b")));
        // Explicit duplicate handling.
        assertNotNull(Expression.merge(
                Expression.DuplicateHandling.RIGHT_MOST, Expression.column("a"), Expression.column("b")));
        // Merging zero expressions is valid and yields an empty struct.
        assertNotNull(Expression.merge());
    }

    @Test
    public void literalTimeAcceptsEveryUnitExceptDays() {
        for (Expression.TimeUnit unit : new Expression.TimeUnit[] {
            Expression.TimeUnit.NANOSECONDS,
            Expression.TimeUnit.MICROSECONDS,
            Expression.TimeUnit.MILLISECONDS,
            Expression.TimeUnit.SECONDS
        }) {
            assertNotNull(Expression.literalTime(3_600L, unit), unit::name);
            assertNotNull(Expression.nullLiteralTime(unit), unit::name);
        }
        RuntimeException exception = assertThrows(
                RuntimeException.class, () -> Expression.literalTime(0L, Expression.TimeUnit.DAYS));
        assertTrue(
                exception.getMessage().contains("Time type does not support time unit"),
                () -> "unexpected message: " + exception.getMessage());
    }

    @Test
    public void literalTimeRejectsSecondsOutsideI32() {
        RuntimeException exception = assertThrows(
                RuntimeException.class,
                () -> Expression.literalTime((long) Integer.MAX_VALUE + 1, Expression.TimeUnit.SECONDS));
        assertTrue(
                exception.getMessage().contains("does not fit in i32"),
                () -> "unexpected message: " + exception.getMessage());
    }

    @Test
    public void literalGeometryBuildsFromWkbPoint() {
        Expression point = Expression.literalGeometry(wkbPoint(1.0, 2.0));
        assertNotNull(Expression.binary(Expression.BinaryOp.EQ, Expression.column("geom"), point));
    }

    @Test
    public void literalGeometryRejectsMalformedWkb() {
        assertThrows(RuntimeException.class, () -> Expression.literalGeometry(new byte[] {1, 2, 3}));
    }

    /** Little-endian WKB for {@code POINT(x y)}. */
    private static byte[] wkbPoint(double x, double y) {
        return ByteBuffer.allocate(21)
                .order(ByteOrder.LITTLE_ENDIAN)
                .put((byte) 1)
                .putInt(1)
                .putDouble(x)
                .putDouble(y)
                .array();
    }
}
