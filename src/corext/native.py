try:
    import corex_native

    NATIVE_AVAILABLE = True

except ImportError:
    NATIVE_AVAILABLE = False


def multiply(a, b):
    if NATIVE_AVAILABLE:
        return corex_native.multiply(a, b)

    return a * b
