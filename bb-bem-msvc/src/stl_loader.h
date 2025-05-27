#ifndef STL_LOADER_H
#define STL_LOADER_H

#include <stdbool.h>

typedef struct {
    float x, y, z;
} stl_vector3_t;

typedef struct {
    stl_vector3_t normal;
    stl_vector3_t v[3];
} stl_facet_t;

typedef struct {
    stl_facet_t* facets;
    int num_facets;
} stl_model_t;

void free_stl_model(stl_model_t* model);

bool load_stl_ascii(const char* filename, stl_model_t* model /* out */);

#endif // STL_LOADER_H
