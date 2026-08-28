find_path(IfcOpenShell_INCLUDE_DIR
  NAMES ifcparse/IfcFile.h
  HINTS
    ${BEEFCLIMATE_IFCOPENSHELL_ROOT}
    ENV IFCOPENSHELL_ROOT
  PATH_SUFFIXES include include/ifcopenshell include/IfcOpenShell
)

find_library(IfcOpenShell_IFCPARSE_LIBRARY
  NAMES IfcParse ifcparse
  HINTS
    ${BEEFCLIMATE_IFCOPENSHELL_ROOT}
    ENV IFCOPENSHELL_ROOT
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IfcOpenShell DEFAULT_MSG IfcOpenShell_INCLUDE_DIR IfcOpenShell_IFCPARSE_LIBRARY)

if(IfcOpenShell_FOUND AND NOT TARGET IfcOpenShell::IfcParse)
  add_library(IfcOpenShell::IfcParse UNKNOWN IMPORTED)
  set_target_properties(IfcOpenShell::IfcParse PROPERTIES
    IMPORTED_LOCATION "${IfcOpenShell_IFCPARSE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${IfcOpenShell_INCLUDE_DIR}"
  )
endif()
