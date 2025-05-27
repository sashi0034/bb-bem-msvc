#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "stl_loader.h"

#define STL_LINE_BUFFER_SIZE 256

void free_stl_model(stl_model_t* model) {
    if (!model) return;

    if (model->facets) {
        free(model->facets);
        model->facets = NULL;
    }

    model->num_facets = 0;
}

static bool stl_detail_read_line(char** out, char* buffer, int max_count, FILE* fp) {
    static const char* s_empty = "";
    (*out)[0] = s_empty[0];

    if (!fgets(buffer, max_count, fp)) {
        return false;
    }

    *out = buffer;

    while (**out && isspace((unsigned char)**out)) {
        (*out)++;
    }

    if (**out == '\0') {
        return false;
    }

    return true;
}

bool load_stl_ascii(const char* filename, stl_model_t* model /* out */) {
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
    char line_buffer[STL_LINE_BUFFER_SIZE];
    char* line = line_buffer;

    // Read header (solid ...)
    if (!fgets(line, STL_LINE_BUFFER_SIZE, fp)) {
        fclose(fp);
        return false;
    }

    while (stl_detail_read_line(&line, line_buffer, STL_LINE_BUFFER_SIZE, fp)) {
        if (strncmp(line, "endsolid", 8) == 0) {
            break;
        }

        if (strncmp(line, "facet normal", 12) == 0) {
            stl_facet_t current;

            // Parse normal
            if (sscanf(line, "facet normal %f %f %f", &normal.x, &normal.y, &normal.z) != 3) {
                continue;
            }

            // Read "outer loop"
            while (!stl_detail_read_line(&line, line_buffer, STL_LINE_BUFFER_SIZE, fp)) {
            }

            // Read 3 vertices
            for (int i = 0; i < 3; i++) {
                while (stl_detail_read_line(&line, line_buffer, STL_LINE_BUFFER_SIZE, fp)) {
                    if (strncmp(line, "vertex", 6) == 0) {
                        sscanf(line, "vertex %f %f %f",
                               &current.v[i].x, &current.v[i].y, &current.v[i].z);
                        break;
                    }
                }
            }

            // Read "endloop" and "endfacet"
            int skipCount = 0;
            while (skipCount < 2 && stl_detail_read_line(&line, line_buffer, STL_LINE_BUFFER_SIZE, fp)) {
                if (*line != '\0') {
                    skipCount++;
                }
            }

            // Store facet
            current.normal = normal;
            if (model->num_facets >= capacity) {
                capacity = capacity ? capacity * 2 : 16;
                stl_facet_t* tmp = (stl_facet_t*)realloc(model->facets, capacity * sizeof(stl_facet_t));
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
