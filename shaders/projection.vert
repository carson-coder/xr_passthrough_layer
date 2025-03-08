#version 450
layout(location = 0) in vec2 position;
layout(location = 1) in uint in_eyeIndex;
layout(binding = 0) uniform Transform {
	mat4 mvp;
	float overlayWidth;
	vec2 eyeOffset;
};
layout(location = 0) out noperspective vec3 texCoord;
layout(location = 1) out flat uint eyeIndex;

void main() {
	vec2 pos = (position + eyeOffset) * overlayWidth / 2.0;
	// Change coordinate system: mvp is y up, position is y down
	pos.y = -pos.y;

	vec4 coord = mvp * vec4(pos, 0, 1);
	texCoord = coord.xyz;

	gl_Position = vec4(position, 0, 1);
    eyeIndex = in_eyeIndex;
}
