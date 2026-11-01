#version 430 core

noperspective in vec2 v_distance;
flat in mat2 v_cov2d;
flat in vec4 v_color;

layout(location = 0) out vec4 fragColor;

float computeWeight(vec2 d, mat2 cov2d) {
    float a = cov2d[0][0];
    float b = cov2d[1][0];
    float c = cov2d[1][1];
    float s = max(a * c - b * b, 1e-6);
    float r2 = (c * d.x * d.x - 2.0 * b * d.x * d.y + a * d.y * d.y) / s;
    float weight = exp(-0.5 * r2);
    return weight;
}

void main()
{
    vec2 d = v_distance;

    float w = computeWeight(d, v_cov2d);

    // quad 覆盖约 3 sigma；外部片元无需参与混合。
    if (w < 0.001) {
        discard;
    }

    vec4 color = v_color;
    color.a = color.a * w;
    color.rgb = color.rgb * color.a;

    fragColor = color;
}
