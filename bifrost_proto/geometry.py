"""Discrete Bifrost-grid geometry shared by the resolver and diagnostics."""

LOCATION_SHIFTED_AXES = {
    "center": (),
    "xface": ("x",),
    "yface": ("y",),
    "zface": ("z",),
    "xedge": ("y", "z"),
    "yedge": ("x", "z"),
    "zedge": ("x", "y"),
}

# Operations that move a native stagger location to cell centers.
CENTER_OPERATIONS = {
    "center": (),
    "xface": ("xup",),
    "yface": ("yup",),
    "zface": ("zup",),
    "xedge": ("yup", "zup"),
    "yedge": ("zup", "xup"),
    "zedge": ("xup", "yup"),
}

FACE_VECTOR_LOCATIONS = ("xface", "yface", "zface")
EDGE_VECTOR_LOCATIONS = ("xedge", "yedge", "zedge")

# A compatible curl maps face vectors to edges and edge vectors to faces.
CURL_OUTPUT_LOCATIONS = {
    FACE_VECTOR_LOCATIONS: EDGE_VECTOR_LOCATIONS,
    EDGE_VECTOR_LOCATIONS: FACE_VECTOR_LOCATIONS,
}
