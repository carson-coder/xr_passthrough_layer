#version 450
// Switches:
//
// INPUT_IS_YUYV: when set the input texture is in yuyv pixel format, otherwise
// it is RGB (or is yuyv but converted by the VK_KHR_sampler_ycbcr_conversion extension),
// and `sample_yuyv2` will be no-op.
// UNDISTORT: whether to apply lens undistortion.

layout(binding = 0) uniform sampler2D inputTex;
layout(binding = 1) uniform DistortionParameters {
    // Parameters are per-eye
    // 0 = left, 1 = right

    // Distortion coefficients
    vec4 dcoef[2];
    // Optical center
    vec2 center[2];
    // Focal length in terms of focal divided by sensor_width
    // 0 = left, 1 = right
    vec2 focal[2];
    // Scaling of the output image
    // 0 = left, 1 = right
    vec2 scale[2];
    // Pixel size of the sensor_width
    float sensorSize;
};

// Input coordinates -0.5 ~ 0.5
// relative to the center of the undistorted image,
// this way we align the optical center to the center of the image.
layout(location = 0) in noperspective vec3 texCoord;
// 0 = left, 1 = right
layout(location = 1) in flat uint eyeIndex;

layout(location = 0) out vec4 color;

// stages: yuyv -> rgb -> distortion corrected -> projected

#ifdef INPUT_IS_YUYV
// bt709 -> rgb conversion matrix
const mat3 yuvMatrix = mat3(
	1.164,  1.164, 1.164,
	0.000, -0.392, 2.017,
	1.596, -0.813, 0.000
);

vec4 sample_input(vec2 coord) { // `coord` in `rgb` coord system
	coord = coord + vec2(0.5, 0.5);

	vec2 size = vec2(textureSize(inputTex, 0));
	vec2 tex_coord = vec2(floor(coord.x / 2.0) + 0.5, coord.y + 0.5) / size;
	vec4 yuyv = texture(inputTex, tex_coord);
	vec3 yuv;
	if (mod(coord.x, 2.0) == 0) {
		yuv = vec3(yuyv.xyw);
	} else {
		yuv = vec3(yuyv.zyw);
	}
	yuv -= vec3(0.0625, 0.5, 0.5);
	return vec4(yuvMatrix * yuv, 1.0);
}
#else
vec4 sample_input(vec2 coord) { // `coord` in `rgb` coord system
	vec2 size = vec2(textureSize(inputTex, 0));
	vec2 tex_coord = coord / size;
	vec4 rgb = texture(inputTex, tex_coord);
	return vec4(rgb.rgb, 1.0);
}
#endif

#ifdef UNDISTORT
vec4 undistort(vec2 coord, uint eyeIndex) {
    float texOffsetX = 0.5 * float(eyeIndex);
    coord.x -= texOffsetX;

    vec2 r = coord * scale[eyeIndex] / focal[eyeIndex];
    // Also scale the r so the whole circular region will be included
    // in the output.
    float theta = atan(length(r));
    float theta2 = theta * theta;
    vec4 c = dcoef[eyeIndex];

    theta *= 1 + theta2 * (c.x +
                 theta2 * (c.y +
                 theta2 * (c.z +
                 theta2 * c.w)));
    // Scale r vector to length theta
    vec2 mapped = theta / length(r) * r;
    mapped *= focal[eyeIndex];
    // mapped should now be -0.5~0.5, in inputTex coord
    // move mapped so its centered at `center`
    mapped = mapped + center[eyeIndex];
    // mapped is now 0 ~ 1
    // scale x by 0.5 because inputTex is 2 image side by side
    mapped.x *= 0.5;
    // mapped is now (0~0.5, 0~1.0);

    return sample_input(mapped + vec2(texOffsetX, 0.0));
}
#else
vec4 undistort(vec2 coord, float texOffsetX) {
    return sample_input(coord + vec2(texOffsetX, 0.0));
}
#endif

void main() {
	// Perspective divide here, if we do this in vertex
	// shader the texCoord won't be interpolated correctly
	// because of perspective.
	vec2 tex_coord = texCoord.xy / texCoord.z;
	tex_coord = tex_coord + vec2(0.25, 0.5);

	if (tex_coord.x < 0 || tex_coord.x > 0.5) {
		color = vec4(0.0, 0.0, 0.0, 0.0);
	} else {
		tex_coord = tex_coord;
		tex_coord.y = 1.0 - tex_coord.y;
		color = undistort(tex_coord, eyeIndex);
	}
}
