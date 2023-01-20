


cbuffer cbSsao : register(b0)
{
	float4x4 gProj;
	float4x4 gInvProj;
	float4x4 gProjTex;
	float4   gOffsetVectors[14];

	float4 gBlurWeights[3];

	float2 gInvRenderTargetSize;

	float    gOcclusionRadius;
	float    gOcclusionFadeStart;
	float    gOcclusionFadeEnd;
	float    gSurfaceEpsilon;
};

cbuffer cbRootConstants : register(b1)
{
	bool gHorizontalBlur;
};

Texture2D gNormalMap    : register(t0);
Texture2D gDepthMap     : register(t1);
Texture2D gInputMap : register(t2);

RWTexture2D<float4> gSsaoMap : register(u0);

SamplerState gsamPointClamp : register(s0);
SamplerState gsamLinearClamp : register(s1);
SamplerState gsamDepthMap : register(s2);
SamplerState gsamLinearWrap : register(s3);

static const int gBlurRadius = 5;

float NdcDepthToViewDepth(float z_ndc)
{
	float viewZ = gProj[3][2] / (z_ndc - gProj[2][2]);
	return viewZ;
}

[numthreads(16, 16, 1)]
void CS(int3 dispatchThreadID : SV_DispatchThreadID)
{
	float2 currentPixelTexC = { dispatchThreadID.x * gInvRenderTargetSize.x, dispatchThreadID.y * gInvRenderTargetSize.y };

	float blurWeights[12] =
	{
		gBlurWeights[0].x, gBlurWeights[0].y, gBlurWeights[0].z, gBlurWeights[0].w,
		gBlurWeights[1].x, gBlurWeights[1].y, gBlurWeights[1].z, gBlurWeights[1].w,
		gBlurWeights[2].x, gBlurWeights[2].y, gBlurWeights[2].z, gBlurWeights[2].w,
	};

	float2 texOffset;
	if (gHorizontalBlur)
	{
		texOffset = float2(gInvRenderTargetSize.x, 0.0f);
	}
	else
	{
		texOffset = float2(0.0f, gInvRenderTargetSize.y);
	}

	float4 color = blurWeights[gBlurRadius] * gInputMap.SampleLevel(gsamPointClamp, currentPixelTexC, 0.0f);//Point
	float totalWeight = blurWeights[gBlurRadius];

	float3 centerNormal = gNormalMap.SampleLevel(gsamPointClamp, currentPixelTexC, 0.0f).xyz;
	float centerDepth = NdcDepthToViewDepth(gDepthMap.SampleLevel(gsamDepthMap, currentPixelTexC, 0.0f).r);

	for (float i = -gBlurRadius; i <= gBlurRadius; ++i)
	{
		if (i == 0)
			continue;

		float2 tex = currentPixelTexC + i * texOffset;

		float3 neighborNormal = gNormalMap.SampleLevel(gsamPointClamp, tex, 0.0f).xyz;
		float neighborDepth = NdcDepthToViewDepth(gDepthMap.SampleLevel(gsamDepthMap, tex, 0.0f).r);

		if (dot(neighborNormal, centerNormal) >= 0.8f && abs(neighborDepth - centerDepth) <= 0.2f)
		{
			float weight = blurWeights[i + gBlurRadius];

			color += weight * gInputMap.SampleLevel(gsamPointClamp, tex, 0.0);

			totalWeight += weight;
		}
	}

	gSsaoMap[dispatchThreadID.xy] = color / totalWeight;
}
