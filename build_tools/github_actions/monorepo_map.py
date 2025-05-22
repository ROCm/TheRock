"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
monorepo_map = {
    "projects/rocprim": {
        "flag": "-DTHEROCK_ENABLE_PRIM=ON -DTHEROCK_ENABLE_ALL=OFF",
        "test": "test_rocprim",
    }
}
