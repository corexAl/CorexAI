#include <pybind11/pybind11.h>
#include "corex.hpp"

namespace py = pybind11;

PYBIND11_MODULE(corex_native, m) {
    m.doc() = "COREX native acceleration module";

    m.def(
        "multiply",
        &corex::multiply,
        "Fast native multiplication"
    );
}
