set(SRanipal_DIR
        "${SRanipal_DIR}"
        CACHE
        PATH
        "Directory to search")

find_path(SRanipal_INCLUDE_DIR
        SRanipal.h
        PATHS
        ${SRanipal_DIR}
        PATH_SUFFIXES
        include/SRanipal
        include
        SRanipal
)

find_library(SRanipal_LIBRARY
        SRanipal
        PATHS
        ${SRanipal_DIR}/lib
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(SRanipal DEFAULT_MSG SRanipal_INCLUDE_DIR SRanipal_LIBRARY)

mark_as_advanced(SRanipal_LIBRARY)
mark_as_advanced(SRanipal_INCLUDE_DIR)

