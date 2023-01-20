

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
Texture2D gRandomVecMap : register(t2);

RWTexture2D<float4> gSsaoMap : register(u0);

SamplerState gsamPointClamp : register(s0);
SamplerState gsamLinearClamp : register(s1);
SamplerState gsamDepthMap : register(s2);
SamplerState gsamLinearWrap : register(s3);

static const int gSampleCount = 14;

float OcclusionFunction(float distZ)
{
	float occlusion = 0.0f;
	if (distZ > gSurfaceEpsilon)
	{
		float fadeLength = gOcclusionFadeEnd - gOcclusionFadeStart;
	
		occlusion = saturate((gOcclusionFadeEnd - distZ) / fadeLength);
	}

	return occlusion;
}

float NdcDepthToViewDepth(float z_ndc)
{
	float viewZ = gProj[3][2] / (z_ndc - gProj[2][2]);
	return viewZ;
}

[numthreads(16, 16, 1)]
void CS(int3 dispatchThreadID : SV_DispatchThreadID)
{
	float2 currentPixelTexC = { dispatchThreadID.x * gInvRenderTargetSize.x, dispatchThreadID.y * gInvRenderTargetSize.y };

	float3 n = normalize(gNormalMap.SampleLevel(gsamPointClamp, currentPixelTexC, 0.0f).xyz);
	float pz = gDepthMap.SampleLevel(gsamDepthMap, currentPixelTexC, 0.0f).r;
	pz = NdcDepthToViewDepth(pz);

	float4 posH = float4(2.0f * currentPixelTexC.x - 1.0f, 1.0f - 2.0f * currentPixelTexC.y, 0.0f, 1.0f);
	float4 ph = mul(posH, gInvProj);
	float3 posV = ph.xyz / ph.w;
	float3 p = (pz / posV.z) * posV;

	float3 randVec = 2.0f * gRandomVecMap.SampleLevel(gsamLinearWrap, 4.0f * currentPixelTexC, 0.0f).rgb - 1.0f;


	float occlusionSum = 0.0f;

	for (int i = 0; i < gSampleCount; ++i)
	{
		float3 offset = reflect(gOffsetVectors[i].xyz, randVec);

		float flip = sign(dot(offset, n));

		float3 q = p + flip * gOcclusionRadius * offset;

		float4 projQ = mul(float4(q, 1.0f), gProjTex);
		projQ /= projQ.w;

		float rz = gDepthMap.SampleLevel(gsamDepthMap, projQ.xy, 0.0f).r;
		rz = NdcDepthToViewDepth(rz);

		float3 r = (rz / q.z) * q;

		float distZ = p.z - r.z;
		float dp = max(dot(n, normalize(r - p)), 0.0f);

		float occlusion = dp * OcclusionFunction(distZ);

		occlusionSum += occlusion;
	}

	occlusionSum /= gSampleCount;

	float access = 1.0f - occlusionSum;
	float4 tmp = { saturate(pow(access, 6.0f)), 0.0f, 0.0f, 1.0f };
	gSsaoMap[dispatchThreadID.xy] = tmp;
}
