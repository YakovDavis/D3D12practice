


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

#define N 256
#define CacheSize (N + 2*gBlurRadius)
groupshared float4 gCache[CacheSize];

[numthreads(N, 1, 1)]
void HorzBlurCS(int3 groupThreadID : SV_GroupThreadID, int3 dispatchThreadID : SV_DispatchThreadID)
{
	float2 currentPixelTexC = { dispatchThreadID.x * gInvRenderTargetSize.x, dispatchThreadID.y * gInvRenderTargetSize.y };

	float blurWeights[12] =
	{
		gBlurWeights[0].x, gBlurWeights[0].y, gBlurWeights[0].z, gBlurWeights[0].w,
		gBlurWeights[1].x, gBlurWeights[1].y, gBlurWeights[1].z, gBlurWeights[1].w,
		gBlurWeights[2].x, gBlurWeights[2].y, gBlurWeights[2].z, gBlurWeights[2].w,
	};

	if (groupThreadID.x < gBlurRadius)
	{
		int x = max(dispatchThreadID.x - gBlurRadius, 0);
		gCache[groupThreadID.x] = gInputMap[int2(x, dispatchThreadID.y)];
	}
	if (groupThreadID.x >= N - gBlurRadius)
	{
		int x = min(dispatchThreadID.x + gBlurRadius, gInputMap.Length.x - 1);
		gCache[groupThreadID.x + 2 * gBlurRadius] = gInputMap[int2(x, dispatchThreadID.y)];
	}

	gCache[groupThreadID.x + gBlurRadius] = gInputMap[min(dispatchThreadID.xy, gInputMap.Length.xy - 1)];

	GroupMemoryBarrierWithGroupSync();

	float2 texOffset;
	texOffset = float2(gInvRenderTargetSize.x, 0.0f);

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

		int k = groupThreadID.x + gBlurRadius + i;

		if (dot(neighborNormal, centerNormal) >= 0.8f && abs(neighborDepth - centerDepth) <= 0.2f)
		{
			float weight = blurWeights[i + gBlurRadius];

			//color += weight * gInputMap.SampleLevel(gsamPointClamp, tex, 0.0);
			color += weight * gCache[k];

			totalWeight += weight;
		}
	}

	gSsaoMap[dispatchThreadID.xy] = color / totalWeight;
}

[numthreads(1, N, 1)]
void VertBlurCS(int3 groupThreadID : SV_GroupThreadID, int3 dispatchThreadID : SV_DispatchThreadID)
{
	float2 currentPixelTexC = { dispatchThreadID.x * gInvRenderTargetSize.x, dispatchThreadID.y * gInvRenderTargetSize.y };

	float blurWeights[12] =
	{
		gBlurWeights[0].x, gBlurWeights[0].y, gBlurWeights[0].z, gBlurWeights[0].w,
		gBlurWeights[1].x, gBlurWeights[1].y, gBlurWeights[1].z, gBlurWeights[1].w,
		gBlurWeights[2].x, gBlurWeights[2].y, gBlurWeights[2].z, gBlurWeights[2].w,
	};

	if (groupThreadID.y < gBlurRadius)
	{
		int y = max(dispatchThreadID.y - gBlurRadius, 0);
		gCache[groupThreadID.y] = gInputMap[int2(dispatchThreadID.x, y)];
	}
	if (groupThreadID.y >= N - gBlurRadius)
	{
		int y = min(dispatchThreadID.y + gBlurRadius, gInputMap.Length.y - 1);
		gCache[groupThreadID.y + 2 * gBlurRadius] = gInputMap[int2(dispatchThreadID.x, y)];
	}

	gCache[groupThreadID.y + gBlurRadius] = gInputMap[min(dispatchThreadID.xy, gInputMap.Length.xy - 1)];

	GroupMemoryBarrierWithGroupSync();

	float2 texOffset;
	texOffset = float2(0.0f, gInvRenderTargetSize.y);

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

		int k = groupThreadID.y + gBlurRadius + i;

		if (dot(neighborNormal, centerNormal) >= 0.8f && abs(neighborDepth - centerDepth) <= 0.2f)
		{
			float weight = blurWeights[i + gBlurRadius];

			//color += weight * gInputMap.SampleLevel(gsamPointClamp, tex, 0.0);
			color += weight * gCache[k];

			totalWeight += weight;
		}
	}

	gSsaoMap[dispatchThreadID.xy] = color / totalWeight;
}
