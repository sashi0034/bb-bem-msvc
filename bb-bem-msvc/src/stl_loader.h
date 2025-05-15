#ifndef STL_LOADER_H
#define STL_LOADER_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

#define STL_LINE_BUFFER_SIZE 256

static void free_stl_model(stl_model_t* model) {
    if (!model) return;

    if (model->facets) {
        free(model->facets);
        model->facets = NULL;
    }

    model->num_facets = 0;
}

static char* skip_whitespace(char* str) {
    while (*str && isspace((unsigned char)*str)) {
        str++;
    }

    return str;
}

static bool load_stl_ascii(const char* filename, stl_model_t* model /* out */) {
    memset(model, 0, sizeof(stl_model_t));

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open STL file");
        return false;
    }

    model->facets = NULL;
    model->num_facets = 0;

    int capacity = 0;
    stl_vector3_t normal;
    char line[STL_LINE_BUFFER_SIZE];

    // Read header (solid ...)
    if (!fgets(line, STL_LINE_BUFFER_SIZE, fp)) {
        fclose(fp);
        return false;
    }

    while (fgets(line, STL_LINE_BUFFER_SIZE, fp)) {
        char* p = skip_whitespace(line);
        if (*p == '\0') continue;

        if (strncmp(p, "endsolid", 8) == 0) {
            break;
        }

        if (strncmp(p, "facet normal", 12) == 0) {
            stl_facet_t current;

            // Parse normal
            if (sscanf(p, "facet normal %f %f %f", &normal.x, &normal.y, &normal.z) != 3) {
                continue;
            }

            // Read "outer loop"
            do {
                if (!fgets(line, STL_LINE_BUFFER_SIZE, fp)) break;
                p = skip_whitespace(line);
            } while (*p == '\0');

            // Read 3 vertices
            for (int i = 0; i < 3; i++) {
                while (fgets(line, STL_LINE_BUFFER_SIZE, fp)) {
                    p = skip_whitespace(line);
                    if (*p == '\0') continue;

                    if (strncmp(p, "vertex", 6) == 0) {
                        sscanf(p, "vertex %f %f %f",
                               &current.v[i].x, &current.v[i].y, &current.v[i].z);
                        break;
                    }
                }
            }

            // Read "endloop" and "endfacet"
            int skipCount = 0;
            while (skipCount < 2 && fgets(line, STL_LINE_BUFFER_SIZE, fp)) {
                p = skip_whitespace(line);
                if (*p != '\0') {
                    skipCount++;
                }
            }

            // Store facet
            current.normal = normal;
            if (model->num_facets >= capacity) {
                capacity = capacity ? capacity * 2 : 16;
                stl_facet_t* tmp = realloc(model->facets, capacity * sizeof(stl_facet_t));
                if (!tmp) {
                    perror("Memory allocation failed");
                    free_stl_model(model);
                    fclose(fp);
                    return false;
                }

                model->facets = tmp;
            }

            model->facets[model->num_facets++] = current;
        }
    }

    fclose(fp);
    return true;
}

#endif // STL_LOADER_H
