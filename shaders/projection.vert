#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in uint in_eyeIndex;
layout(binding = 0) uniform Transform {
	mat4 mvp[2];
	float overlayWidth;
	vec2 eyeOffset;
};
layout(location = 1) out flat uint eyeIndex;

void main() {
    vec4 pos = mvp[in_eyeIndex] * vec4(position, 1) / vec4(2.0, 1.0, 1.0, 1.0);
    // Change coordinate system: mvp is y up, position is y down
    pos.y = -pos.y;
    gl_Position = pos + vec4(0.5 * float(in_eyeIndex), 0.0, 0.0, 0.0) * pos.w;
    eyeIndex = in_eyeIndex;
}
