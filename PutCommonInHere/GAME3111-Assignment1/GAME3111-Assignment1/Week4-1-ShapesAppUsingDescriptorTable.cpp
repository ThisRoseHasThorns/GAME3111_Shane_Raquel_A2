//***************************************************************************************
// ShapesApp.cpp 
//
// Hold down '1' key to view scene in wireframe mode.
//***************************************************************************************

#include "../../Common/d3dApp.h"
#include "../../Common/MathHelper.h"
#include "../../Common/UploadBuffer.h"
#include "../../Common/GeometryGenerator.h"
#include "FrameResource.h"

using Microsoft::WRL::ComPtr;
using namespace DirectX;
using namespace DirectX::PackedVector;

const int gNumFrameResources = 3;

// Lightweight structure stores parameters to draw a shape.  This will
// vary from app-to-app.
struct RenderItem
{
	RenderItem() = default;

    // World matrix of the shape that describes the object's local space
    // relative to the world space, which defines the position, orientation,
    // and scale of the object in the world.
    XMFLOAT4X4 World = MathHelper::Identity4x4();

	// Dirty flag indicating the object data has changed and we need to update the constant buffer.
	// Because we have an object cbuffer for each FrameResource, we have to apply the
	// update to each FrameResource.  Thus, when we modify obect data we should set 
	// NumFramesDirty = gNumFrameResources so that each frame resource gets the update.
	int NumFramesDirty = gNumFrameResources;

	// Index into GPU constant buffer corresponding to the ObjectCB for this render item.
	UINT ObjCBIndex = -1;

	MeshGeometry* Geo = nullptr;

    // Primitive topology.
    D3D12_PRIMITIVE_TOPOLOGY PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

    // DrawIndexedInstanced parameters.
    UINT IndexCount = 0;
    UINT StartIndexLocation = 0;
    int BaseVertexLocation = 0;
};

class ShapesApp : public D3DApp
{
public:
    ShapesApp(HINSTANCE hInstance);
    ShapesApp(const ShapesApp& rhs) = delete;
    ShapesApp& operator=(const ShapesApp& rhs) = delete;
    ~ShapesApp();

    virtual bool Initialize()override;

private:
    virtual void OnResize()override;
    virtual void Update(const GameTimer& gt)override;
    virtual void Draw(const GameTimer& gt)override;

    virtual void OnMouseDown(WPARAM btnState, int x, int y)override;
    virtual void OnMouseUp(WPARAM btnState, int x, int y)override;
    virtual void OnMouseMove(WPARAM btnState, int x, int y)override;

    void OnKeyboardInput(const GameTimer& gt);
	void UpdateCamera(const GameTimer& gt);
	void UpdateObjectCBs(const GameTimer& gt);
	void UpdateMainPassCB(const GameTimer& gt);

    void BuildDescriptorHeaps();
    void BuildConstantBufferViews();
    void BuildRootSignature();
    void BuildShadersAndInputLayout();
    void BuildShapeGeometry();
    void BuildPSOs();
    void BuildFrameResources();
    void BuildRenderItems();
    void DrawRenderItems(ID3D12GraphicsCommandList* cmdList, const std::vector<RenderItem*>& ritems);
 
private:

    std::vector<std::unique_ptr<FrameResource>> mFrameResources;
    FrameResource* mCurrFrameResource = nullptr;
    int mCurrFrameResourceIndex = 0;

    ComPtr<ID3D12RootSignature> mRootSignature = nullptr;
    ComPtr<ID3D12DescriptorHeap> mCbvHeap = nullptr;

	ComPtr<ID3D12DescriptorHeap> mSrvDescriptorHeap = nullptr;

	std::unordered_map<std::string, std::unique_ptr<MeshGeometry>> mGeometries;
	std::unordered_map<std::string, ComPtr<ID3DBlob>> mShaders;
    std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> mPSOs;

    std::vector<D3D12_INPUT_ELEMENT_DESC> mInputLayout;

	// List of all the render items.
	std::vector<std::unique_ptr<RenderItem>> mAllRitems;

	// Render items divided by PSO.
	std::vector<RenderItem*> mOpaqueRitems;

    PassConstants mMainPassCB;

    UINT mPassCbvOffset = 0;

    bool mIsWireframe = false;

	XMFLOAT3 mEyePos = { 0.0f, 0.0f, 0.0f };
	XMFLOAT4X4 mView = MathHelper::Identity4x4();
	XMFLOAT4X4 mProj = MathHelper::Identity4x4();

    float mTheta = 1.5f*XM_PI;
    float mPhi = 0.2f*XM_PI;
    float mRadius = 15.0f;

    POINT mLastMousePos;
};

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance,
    PSTR cmdLine, int showCmd)
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    try
    {
        ShapesApp theApp(hInstance);
        if(!theApp.Initialize())
            return 0;

        return theApp.Run();
    }
    catch(DxException& e)
    {
        MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
        return 0;
    }
}

ShapesApp::ShapesApp(HINSTANCE hInstance)
    : D3DApp(hInstance)
{
}

ShapesApp::~ShapesApp()
{
    if(md3dDevice != nullptr)
        FlushCommandQueue();
}

bool ShapesApp::Initialize()
{
    if(!D3DApp::Initialize())
        return false;

    // Reset the command list to prep for initialization commands.
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));

    BuildRootSignature();
    BuildShadersAndInputLayout();
    BuildShapeGeometry();
    BuildRenderItems();
    BuildFrameResources();
    BuildDescriptorHeaps();
    BuildConstantBufferViews();
    BuildPSOs();

    // Execute the initialization commands.
    ThrowIfFailed(mCommandList->Close());
    ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    // Wait until initialization is complete.
    FlushCommandQueue();

    return true;
}
 
void ShapesApp::OnResize()
{
    D3DApp::OnResize();

    // The window resized, so update the aspect ratio and recompute the projection matrix.
    XMMATRIX P = XMMatrixPerspectiveFovLH(0.25f*MathHelper::Pi, AspectRatio(), 1.0f, 1000.0f);
    XMStoreFloat4x4(&mProj, P);
}

void ShapesApp::Update(const GameTimer& gt)
{
    OnKeyboardInput(gt);
	UpdateCamera(gt);

    // Cycle through the circular frame resource array.
    mCurrFrameResourceIndex = (mCurrFrameResourceIndex + 1) % gNumFrameResources;
    mCurrFrameResource = mFrameResources[mCurrFrameResourceIndex].get();

    // Has the GPU finished processing the commands of the current frame resource?
    // If not, wait until the GPU has completed commands up to this fence point.
    if(mCurrFrameResource->Fence != 0 && mFence->GetCompletedValue() < mCurrFrameResource->Fence)
    {
        HANDLE eventHandle = CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
        ThrowIfFailed(mFence->SetEventOnCompletion(mCurrFrameResource->Fence, eventHandle));
        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }

	UpdateObjectCBs(gt);
	UpdateMainPassCB(gt);
}

void ShapesApp::Draw(const GameTimer& gt)
{
    auto cmdListAlloc = mCurrFrameResource->CmdListAlloc;

    // Reuse the memory associated with command recording.
    // We can only reset when the associated command lists have finished execution on the GPU.
    ThrowIfFailed(cmdListAlloc->Reset());

    // A command list can be reset after it has been added to the command queue via ExecuteCommandList.
    // Reusing the command list reuses memory.
    if(mIsWireframe)
    {
        ThrowIfFailed(mCommandList->Reset(cmdListAlloc.Get(), mPSOs["opaque_wireframe"].Get()));
    }
    else
    {
        ThrowIfFailed(mCommandList->Reset(cmdListAlloc.Get(), mPSOs["opaque"].Get()));
    }

    mCommandList->RSSetViewports(1, &mScreenViewport);
    mCommandList->RSSetScissorRects(1, &mScissorRect);

    // Indicate a state transition on the resource usage.
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
		D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

    // Clear the back buffer and depth buffer.
    mCommandList->ClearRenderTargetView(CurrentBackBufferView(), Colors::LightSteelBlue, 0, nullptr);
    mCommandList->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

    // Specify the buffers we are going to render to.
    mCommandList->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());

    ID3D12DescriptorHeap* descriptorHeaps[] = { mCbvHeap.Get() };
    mCommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

	mCommandList->SetGraphicsRootSignature(mRootSignature.Get());

    int passCbvIndex = mPassCbvOffset + mCurrFrameResourceIndex;
    auto passCbvHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(mCbvHeap->GetGPUDescriptorHandleForHeapStart());
    passCbvHandle.Offset(passCbvIndex, mCbvSrvUavDescriptorSize);
    mCommandList->SetGraphicsRootDescriptorTable(1, passCbvHandle);

    DrawRenderItems(mCommandList.Get(), mOpaqueRitems);

    // Indicate a state transition on the resource usage.
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
		D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

    // Done recording commands.
    ThrowIfFailed(mCommandList->Close());

    // Add the command list to the queue for execution.
    ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    // Swap the back and front buffers
    ThrowIfFailed(mSwapChain->Present(0, 0));
	mCurrBackBuffer = (mCurrBackBuffer + 1) % SwapChainBufferCount;

    // Advance the fence value to mark commands up to this fence point.
    mCurrFrameResource->Fence = ++mCurrentFence;
    
    // Add an instruction to the command queue to set a new fence point. 
    // Because we are on the GPU timeline, the new fence point won't be 
    // set until the GPU finishes processing all the commands prior to this Signal().
    mCommandQueue->Signal(mFence.Get(), mCurrentFence);
}

void ShapesApp::OnMouseDown(WPARAM btnState, int x, int y)
{
    mLastMousePos.x = x;
    mLastMousePos.y = y;

    SetCapture(mhMainWnd);
}

void ShapesApp::OnMouseUp(WPARAM btnState, int x, int y)
{
    ReleaseCapture();
}

void ShapesApp::OnMouseMove(WPARAM btnState, int x, int y)
{
    if((btnState & MK_LBUTTON) != 0)
    {
        // Make each pixel correspond to a quarter of a degree.
        float dx = XMConvertToRadians(0.25f*static_cast<float>(x - mLastMousePos.x));
        float dy = XMConvertToRadians(0.25f*static_cast<float>(y - mLastMousePos.y));

        // Update angles based on input to orbit camera around box.
        mTheta += dx;
        mPhi += dy;

        // Restrict the angle mPhi.
        mPhi = MathHelper::Clamp(mPhi, 0.1f, MathHelper::Pi - 0.1f);
    }
    else if((btnState & MK_RBUTTON) != 0)
    {
        // Make each pixel correspond to 0.2 unit in the scene.
        float dx = 0.05f*static_cast<float>(x - mLastMousePos.x);
        float dy = 0.05f*static_cast<float>(y - mLastMousePos.y);

        // Update the camera radius based on input.
        mRadius += dx - dy;

        // Restrict the radius.
        mRadius = MathHelper::Clamp(mRadius, 5.0f, 150.0f);
    }

    mLastMousePos.x = x;
    mLastMousePos.y = y;
}
 
void ShapesApp::OnKeyboardInput(const GameTimer& gt)
{
    if(GetAsyncKeyState('1') & 0x8000)
        mIsWireframe = true;
    else
        mIsWireframe = false;
}
 
void ShapesApp::UpdateCamera(const GameTimer& gt)
{
	// Convert Spherical to Cartesian coordinates.
	mEyePos.x = mRadius*sinf(mPhi)*cosf(mTheta);
	mEyePos.z = mRadius*sinf(mPhi)*sinf(mTheta);
	mEyePos.y = mRadius*cosf(mPhi);

	// Build the view matrix.
	XMVECTOR pos = XMVectorSet(mEyePos.x, mEyePos.y, mEyePos.z, 1.0f);
	XMVECTOR target = XMVectorZero();
	XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

	XMMATRIX view = XMMatrixLookAtLH(pos, target, up);
	XMStoreFloat4x4(&mView, view);
}

void ShapesApp::UpdateObjectCBs(const GameTimer& gt)
{
	auto currObjectCB = mCurrFrameResource->ObjectCB.get();
	for(auto& e : mAllRitems)
	{
		// Only update the cbuffer data if the constants have changed.  
		// This needs to be tracked per frame resource.
		if(e->NumFramesDirty > 0)
		{
			XMMATRIX world = XMLoadFloat4x4(&e->World);

			ObjectConstants objConstants;
			XMStoreFloat4x4(&objConstants.World, XMMatrixTranspose(world));

			currObjectCB->CopyData(e->ObjCBIndex, objConstants);

			// Next FrameResource need to be updated too.
			e->NumFramesDirty--;
		}
	}
}

void ShapesApp::UpdateMainPassCB(const GameTimer& gt)
{
	XMMATRIX view = XMLoadFloat4x4(&mView);
	XMMATRIX proj = XMLoadFloat4x4(&mProj);

	XMMATRIX viewProj = XMMatrixMultiply(view, proj);
	XMMATRIX invView = XMMatrixInverse(&XMMatrixDeterminant(view), view);
	XMMATRIX invProj = XMMatrixInverse(&XMMatrixDeterminant(proj), proj);
	XMMATRIX invViewProj = XMMatrixInverse(&XMMatrixDeterminant(viewProj), viewProj);

	XMStoreFloat4x4(&mMainPassCB.View, XMMatrixTranspose(view));
	XMStoreFloat4x4(&mMainPassCB.InvView, XMMatrixTranspose(invView));
	XMStoreFloat4x4(&mMainPassCB.Proj, XMMatrixTranspose(proj));
	XMStoreFloat4x4(&mMainPassCB.InvProj, XMMatrixTranspose(invProj));
	XMStoreFloat4x4(&mMainPassCB.ViewProj, XMMatrixTranspose(viewProj));
	XMStoreFloat4x4(&mMainPassCB.InvViewProj, XMMatrixTranspose(invViewProj));
	mMainPassCB.EyePosW = mEyePos;
	mMainPassCB.RenderTargetSize = XMFLOAT2((float)mClientWidth, (float)mClientHeight);
	mMainPassCB.InvRenderTargetSize = XMFLOAT2(1.0f / mClientWidth, 1.0f / mClientHeight);
	mMainPassCB.NearZ = 1.0f;
	mMainPassCB.FarZ = 1000.0f;
	mMainPassCB.TotalTime = gt.TotalTime();
	mMainPassCB.DeltaTime = gt.DeltaTime();

	auto currPassCB = mCurrFrameResource->PassCB.get();
	currPassCB->CopyData(0, mMainPassCB);
}

//If we have 3 frame resources and n render items, then we have three 3n object constant
//buffers and 3 pass constant buffers.Hence we need 3(n + 1) constant buffer views(CBVs).
//Thus we will need to modify our CBV heap to include the additional descriptors :

void ShapesApp::BuildDescriptorHeaps()
{
    UINT objCount = (UINT)mOpaqueRitems.size();

    // Need a CBV descriptor for each object for each frame resource,
    // +1 for the perPass CBV for each frame resource.
    UINT numDescriptors = (objCount+1) * gNumFrameResources;

    // Save an offset to the start of the pass CBVs.  These are the last 3 descriptors.
    mPassCbvOffset = objCount * gNumFrameResources;

    D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc;
    cbvHeapDesc.NumDescriptors = numDescriptors;
    cbvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    cbvHeapDesc.NodeMask = 0;
    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&cbvHeapDesc,
        IID_PPV_ARGS(&mCbvHeap)));
}

//assuming we have n renter items, we can populate the CBV heap with the following code where descriptors 0 to n-
//1 contain the object CBVs for the 0th frame resource, descriptors n to 2n−1 contains the
//object CBVs for 1st frame resource, descriptors 2n to 3n−1 contain the objects CBVs for
//the 2nd frame resource, and descriptors 3n, 3n + 1, and 3n + 2 contain the pass CBVs for the
//0th, 1st, and 2nd frame resource
void ShapesApp::BuildConstantBufferViews()
{
    UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));

    UINT objCount = (UINT)mOpaqueRitems.size();

    // Need a CBV descriptor for each object for each frame resource.
    for(int frameIndex = 0; frameIndex < gNumFrameResources; ++frameIndex)
    {
        auto objectCB = mFrameResources[frameIndex]->ObjectCB->Resource();
        for(UINT i = 0; i < objCount; ++i)
        {
            D3D12_GPU_VIRTUAL_ADDRESS cbAddress = objectCB->GetGPUVirtualAddress();

            // Offset to the ith object constant buffer in the buffer.
            cbAddress += i*objCBByteSize;

            // Offset to the object cbv in the descriptor heap.
            int heapIndex = frameIndex*objCount + i;

			//we can get a handle to the first descriptor in a heap with the ID3D12DescriptorHeap::GetCPUDescriptorHandleForHeapStart
            auto handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(mCbvHeap->GetCPUDescriptorHandleForHeapStart());

			//our heap has more than one descriptor,we need to know the size to increment in the heap to get to the next descriptor
			//This is hardware specific, so we have to query this information from the device, and it depends on
			//the heap type.Recall that our D3DApp class caches this information: 	mCbvSrvUavDescriptorSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            handle.Offset(heapIndex, mCbvSrvUavDescriptorSize);

            D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
            cbvDesc.BufferLocation = cbAddress;
            cbvDesc.SizeInBytes = objCBByteSize;

            md3dDevice->CreateConstantBufferView(&cbvDesc, handle);
        }
    }

    UINT passCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(PassConstants));

    // Last three descriptors are the pass CBVs for each frame resource.
    for(int frameIndex = 0; frameIndex < gNumFrameResources; ++frameIndex)
    {
        auto passCB = mFrameResources[frameIndex]->PassCB->Resource();
        D3D12_GPU_VIRTUAL_ADDRESS cbAddress = passCB->GetGPUVirtualAddress();

        // Offset to the pass cbv in the descriptor heap.
        int heapIndex = mPassCbvOffset + frameIndex;
        auto handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(mCbvHeap->GetCPUDescriptorHandleForHeapStart());
        handle.Offset(heapIndex, mCbvSrvUavDescriptorSize);

        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
        cbvDesc.BufferLocation = cbAddress;
        cbvDesc.SizeInBytes = passCBByteSize;
        
        md3dDevice->CreateConstantBufferView(&cbvDesc, handle);
    }
}

//A root signature defines what resources need to be bound to the pipeline before issuing a draw call and
//how those resources get mapped to shader input registers. there is a limit of 64 DWORDs that can be put in a root signature.
void ShapesApp::BuildRootSignature()
{
    CD3DX12_DESCRIPTOR_RANGE cbvTable0;
    cbvTable0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);

    CD3DX12_DESCRIPTOR_RANGE cbvTable1;
    cbvTable1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1);

	// Root parameter can be a table, root descriptor or root constants.
	CD3DX12_ROOT_PARAMETER slotRootParameter[2];

	// Create root CBVs.
    slotRootParameter[0].InitAsDescriptorTable(1, &cbvTable0);
    slotRootParameter[1].InitAsDescriptorTable(1, &cbvTable1);

	// A root signature is an array of root parameters.
	CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(2, slotRootParameter, 0, nullptr, 
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	// create a root signature with a single slot which points to a descriptor range consisting of a single constant buffer
	ComPtr<ID3DBlob> serializedRootSig = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;
	HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
		serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());

	if(errorBlob != nullptr)
	{
		::OutputDebugStringA((char*)errorBlob->GetBufferPointer());
	}
	ThrowIfFailed(hr);

	ThrowIfFailed(md3dDevice->CreateRootSignature(
		0,
		serializedRootSig->GetBufferPointer(),
		serializedRootSig->GetBufferSize(),
		IID_PPV_ARGS(mRootSignature.GetAddressOf())));
}

void ShapesApp::BuildShadersAndInputLayout()
{
	mShaders["standardVS"] = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "VS", "vs_5_1");
	mShaders["opaquePS"] = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "PS", "ps_5_1");
	
    mInputLayout =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
}

void ShapesApp::BuildShapeGeometry()
{
    GeometryGenerator geoGen;
    /*GeometryGenerator::MeshData box = geoGen.CreateBox(1.5f, 0.5f, 1.5f, 3);
	GeometryGenerator::MeshData grid = geoGen.CreateGrid(100.0f, 100.0f, 10, 10);
	GeometryGenerator::MeshData sphere = geoGen.CreateSphere(0.5f, 20, 20);
	GeometryGenerator::MeshData cylinder = geoGen.CreateCylinder(0.5f, 0.3f, 3.0f, 20, 20);
    GeometryGenerator::MeshData cone = geoGen.CreateCone(1.0f, 5.0f, false, 20, 20);
    GeometryGenerator::MeshData leftWall = geoGen.CreateBox(1.5f, 0.5f, 1.5f, 3);
    GeometryGenerator::MeshData rightWall = geoGen.CreateBox(1.5f, 0.5f, 1.5f, 3);
    GeometryGenerator::MeshData backWall = geoGen.CreateBox(1.5f, 0.5f, 1.5f, 3);
    GeometryGenerator::MeshData frontWallLeft = geoGen.CreateBox(1.5f, 0.5f, 1.5f, 3);
    GeometryGenerator::MeshData frontWallTop = geoGen.CreateBox(1.5f, 0.5f, 1.5f, 3);
    GeometryGenerator::MeshData merlin = geoGen.CreateBox(1.5f, 0.5f, 1.5f, 3);
    GeometryGenerator::MeshData floor = geoGen.CreateBox(1.0f, 1.0f, 1.0f, 3);*/
    GeometryGenerator::MeshData grid = geoGen.CreateGrid(100.0f, 100.0f, 10, 10);
    GeometryGenerator::MeshData wall = geoGen.CreateBox(1.0f, 1.0f, 1.0f, 3);
    GeometryGenerator::MeshData tower = geoGen.CreateCylinder(1.0f, 0.8f, 1.0f, 20, 20);
    GeometryGenerator::MeshData cone = geoGen.CreateCone(1.0f, 1.0f, false, 20, 20);
    GeometryGenerator::MeshData floor = geoGen.CreateBox(1.0f, 1.0f, 1.0f, 3);
    GeometryGenerator::MeshData wedge = geoGen.CreateWedge(1.0f, 1.0f, 1.0f, 3);
    GeometryGenerator::MeshData pyramid = geoGen.CreatePyramid(1.0f, 1.0f, 1.0f);
    GeometryGenerator::MeshData diamond = geoGen.CreateDiamond(1.0f, 1.0f, 2.0f);
   

	//
	// We are concatenating all the geometry into one big vertex/index buffer.  So
	// define the regions in the buffer each submesh covers.
	//

	// Cache the vertex offsets to each object in the concatenated vertex buffer.
	/*UINT boxVertexOffset = 0;
	UINT gridVertexOffset = (UINT)box.Vertices.size();
	UINT sphereVertexOffset = gridVertexOffset + (UINT)grid.Vertices.size();
	UINT cylinderVertexOffset = sphereVertexOffset + (UINT)sphere.Vertices.size();
    UINT coneVertexOffset = cylinderVertexOffset + (UINT)cylinder.Vertices.size();*/
    UINT gridVertexOffset = 0;
    UINT wallVertexOffset = gridVertexOffset + (UINT)grid.Vertices.size();
    UINT towerVertexOffset = wallVertexOffset + (UINT)wall.Vertices.size();
    UINT coneVertexOffset = towerVertexOffset + (UINT)tower.Vertices.size();
    UINT floorVertexOffset = coneVertexOffset + (UINT)cone.Vertices.size();
    UINT wedgeVertexOffset = floorVertexOffset + (UINT)floor.Vertices.size();
    UINT pyramidVertexOffset = wedgeVertexOffset + (UINT)wedge.Vertices.size();
    UINT diamondVertexOffset = pyramidVertexOffset + (UINT)pyramid.Vertices.size();

	// Cache the starting index for each object in the concatenated index buffer.
	/*UINT boxIndexOffset = 0;
	UINT gridIndexOffset = (UINT)box.Indices32.size();
	UINT sphereIndexOffset = gridIndexOffset + (UINT)grid.Indices32.size();
	UINT cylinderIndexOffset = sphereIndexOffset + (UINT)sphere.Indices32.size();
    UINT coneIndexOffset = cylinderIndexOffset + (UINT)cylinder.Indices32.size();*/
    UINT gridIndexOffset = 0;
    UINT wallIndexOffset = gridIndexOffset + (UINT)grid.Indices32.size();
    UINT towerIndexOffset = wallIndexOffset + (UINT)wall.Indices32.size();
    UINT coneIndexOffset = towerIndexOffset + (UINT)tower.Indices32.size();
    UINT floorIndexOffset = coneIndexOffset + (UINT)cone.Indices32.size();
    UINT wedgeIndexOffset = floorIndexOffset + (UINT)floor.Indices32.size();
    UINT pyramidIndexOffset = wedgeIndexOffset + (UINT)wedge.Indices32.size();
    UINT diamondIndexOffset = pyramidIndexOffset + (UINT)pyramid.Indices32.size();

    // Define the SubmeshGeometry that cover different 
    // regions of the vertex/index buffers.

    SubmeshGeometry gridSubmesh;
    gridSubmesh.IndexCount = (UINT)grid.Indices32.size();
    gridSubmesh.StartIndexLocation = gridIndexOffset;
    gridSubmesh.BaseVertexLocation = gridVertexOffset;

    SubmeshGeometry wallSubmesh;
    wallSubmesh.IndexCount = (UINT)wall.Indices32.size();
    wallSubmesh.StartIndexLocation = wallIndexOffset;
    wallSubmesh.BaseVertexLocation = wallVertexOffset;

    SubmeshGeometry towerSubmesh;
    towerSubmesh.IndexCount = (UINT)tower.Indices32.size();
    towerSubmesh.StartIndexLocation = towerIndexOffset;
    towerSubmesh.BaseVertexLocation = towerVertexOffset;

    SubmeshGeometry coneSubmesh;
    coneSubmesh.IndexCount = (UINT)cone.Indices32.size();
    coneSubmesh.StartIndexLocation = coneIndexOffset;
    coneSubmesh.BaseVertexLocation = coneVertexOffset;

    SubmeshGeometry floorSubmesh;
    floorSubmesh.IndexCount = (UINT)floor.Indices32.size();
    floorSubmesh.StartIndexLocation = floorIndexOffset;
    floorSubmesh.BaseVertexLocation = floorVertexOffset;

    SubmeshGeometry wedgeSubmesh;
    wedgeSubmesh.IndexCount = (UINT)wedge.Indices32.size();
    wedgeSubmesh.StartIndexLocation = wedgeIndexOffset;
    wedgeSubmesh.BaseVertexLocation = wedgeVertexOffset;

    SubmeshGeometry pyramidSubmesh;
    pyramidSubmesh.IndexCount = (UINT)pyramid.Indices32.size();
    pyramidSubmesh.StartIndexLocation = pyramidIndexOffset;
    pyramidSubmesh.BaseVertexLocation = pyramidVertexOffset;

    SubmeshGeometry diamondSubmesh;
    diamondSubmesh.IndexCount = (UINT)diamond.Indices32.size();
    diamondSubmesh.StartIndexLocation = diamondIndexOffset;
    diamondSubmesh.BaseVertexLocation = diamondVertexOffset;

	/*SubmeshGeometry boxSubmesh;
	boxSubmesh.IndexCount = (UINT)box.Indices32.size();
	boxSubmesh.StartIndexLocation = boxIndexOffset;
	boxSubmesh.BaseVertexLocation = boxVertexOffset;*/

	

	/*SubmeshGeometry sphereSubmesh;
	sphereSubmesh.IndexCount = (UINT)sphere.Indices32.size();
	sphereSubmesh.StartIndexLocation = sphereIndexOffset;
	sphereSubmesh.BaseVertexLocation = sphereVertexOffset;*/

	/*SubmeshGeometry cylinderSubmesh;
	cylinderSubmesh.IndexCount = (UINT)cylinder.Indices32.size();
	cylinderSubmesh.StartIndexLocation = cylinderIndexOffset;
	cylinderSubmesh.BaseVertexLocation = cylinderVertexOffset;*/

    /*SubmeshGeometry coneSubmesh;
    coneSubmesh.IndexCount = (UINT)cone.Indices32.size();
    coneSubmesh.StartIndexLocation = coneIndexOffset;
    coneSubmesh.BaseVertexLocation = coneVertexOffset;*/

	//
	// Extract the vertex elements we are interested in and pack the
	// vertices of all the meshes into one vertex buffer.
	//

	/*auto totalVertexCount =
		box.Vertices.size() +
		grid.Vertices.size() +
		sphere.Vertices.size() +
		cylinder.Vertices.size() + 
        cone.Vertices.size();*/
    auto totalVertexCount =
        grid.Vertices.size() +
        wall.Vertices.size() +
        tower.Vertices.size() +
        cone.Vertices.size() +
        floor.Vertices.size() +
        wedge.Vertices.size() +
        pyramid.Vertices.size() +
        diamond.Vertices.size();
    

	std::vector<Vertex> vertices(totalVertexCount);

	UINT k = 0;
    for (size_t i = 0; i < grid.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = grid.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::ForestGreen);
    }

    for (size_t i = 0; i < wall.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = wall.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::Navy);
    }

    for (size_t i = 0; i < tower.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = tower.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::SteelBlue);
    }

    for (size_t i = 0; i < cone.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = cone.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::MediumPurple);
    }

    for (size_t i = 0; i < floor.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = floor.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::SlateGray);
    }

    for (size_t i = 0; i < wedge.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = wedge.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::SandyBrown);
    }

    for (size_t i = 0; i < pyramid.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = pyramid.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::Goldenrod);
    }

    for (size_t i = 0; i < diamond.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = diamond.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::Black);
    }

	/*for(size_t i = 0; i < box.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = box.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::Navy);
	}

	for(size_t i = 0; i < grid.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = grid.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::ForestGreen);
	}

	for(size_t i = 0; i < sphere.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = sphere.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::Crimson);
	}

	for(size_t i = 0; i < cylinder.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = cylinder.Vertices[i].Position;
		vertices[k].Color = XMFLOAT4(DirectX::Colors::SteelBlue);
	}

    for (size_t i = 0; i < cone.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = cone.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::MediumPurple);
    }*/


	std::vector<std::uint16_t> indices;
    indices.insert(indices.end(), std::begin(grid.GetIndices16()), std::end(grid.GetIndices16()));
    indices.insert(indices.end(), std::begin(wall.GetIndices16()), std::end(wall.GetIndices16()));
    indices.insert(indices.end(), std::begin(tower.GetIndices16()), std::end(tower.GetIndices16()));
    indices.insert(indices.end(), std::begin(cone.GetIndices16()), std::end(cone.GetIndices16()));
    indices.insert(indices.end(), std::begin(floor.GetIndices16()), std::end(floor.GetIndices16()));
    indices.insert(indices.end(), std::begin(wedge.GetIndices16()), std::end(wedge.GetIndices16()));
    indices.insert(indices.end(), std::begin(pyramid.GetIndices16()), std::end(pyramid.GetIndices16()));
    indices.insert(indices.end(), std::begin(diamond.GetIndices16()), std::end(diamond.GetIndices16()));
	/*indices.insert(indices.end(), std::begin(box.GetIndices16()), std::end(box.GetIndices16()));
	indices.insert(indices.end(), std::begin(grid.GetIndices16()), std::end(grid.GetIndices16()));
	indices.insert(indices.end(), std::begin(sphere.GetIndices16()), std::end(sphere.GetIndices16()));
    indices.insert(indices.end(), std::begin(cylinder.GetIndices16()), std::end(cylinder.GetIndices16()));
    indices.insert(indices.end(), std::begin(cone.GetIndices16()), std::end(cone.GetIndices16()));*/

    const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);
    const UINT ibByteSize = (UINT)indices.size()  * sizeof(std::uint16_t);

	auto geo = std::make_unique<MeshGeometry>();
	geo->Name = "shapeGeo";

	ThrowIfFailed(D3DCreateBlob(vbByteSize, &geo->VertexBufferCPU));
	CopyMemory(geo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

	ThrowIfFailed(D3DCreateBlob(ibByteSize, &geo->IndexBufferCPU));
	CopyMemory(geo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

	geo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), vertices.data(), vbByteSize, geo->VertexBufferUploader);

	geo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), indices.data(), ibByteSize, geo->IndexBufferUploader);

	geo->VertexByteStride = sizeof(Vertex);
	geo->VertexBufferByteSize = vbByteSize;
	geo->IndexFormat = DXGI_FORMAT_R16_UINT;
	geo->IndexBufferByteSize = ibByteSize;

    geo->DrawArgs["grid"] = gridSubmesh;
    geo->DrawArgs["wall"] = wallSubmesh;
    geo->DrawArgs["tower"] = towerSubmesh;
    geo->DrawArgs["cone"] = coneSubmesh;
    geo->DrawArgs["floor"] = floorSubmesh;
    geo->DrawArgs["wedge"] = wedgeSubmesh;
    geo->DrawArgs["pyramid"] = pyramidSubmesh;
    geo->DrawArgs["diamond"] = diamondSubmesh;
	/*geo->DrawArgs["box"] = boxSubmesh;
	geo->DrawArgs["grid"] = gridSubmesh;
	geo->DrawArgs["sphere"] = sphereSubmesh;
	geo->DrawArgs["cylinder"] = cylinderSubmesh;
    geo->DrawArgs["cone"] = coneSubmesh;
    geo->DrawArgs["merlin"] = boxSubmesh;*/

	mGeometries[geo->Name] = std::move(geo);
}

void ShapesApp::BuildPSOs()
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC opaquePsoDesc;

	//
	// PSO for opaque objects.
	//
    ZeroMemory(&opaquePsoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
	opaquePsoDesc.InputLayout = { mInputLayout.data(), (UINT)mInputLayout.size() };
	opaquePsoDesc.pRootSignature = mRootSignature.Get();
	opaquePsoDesc.VS = 
	{ 
		reinterpret_cast<BYTE*>(mShaders["standardVS"]->GetBufferPointer()), 
		mShaders["standardVS"]->GetBufferSize()
	};
	opaquePsoDesc.PS = 
	{ 
		reinterpret_cast<BYTE*>(mShaders["opaquePS"]->GetBufferPointer()),
		mShaders["opaquePS"]->GetBufferSize()
	};
	opaquePsoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    opaquePsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
	opaquePsoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	opaquePsoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	opaquePsoDesc.SampleMask = UINT_MAX;
	opaquePsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	opaquePsoDesc.NumRenderTargets = 1;
	opaquePsoDesc.RTVFormats[0] = mBackBufferFormat;
	opaquePsoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
	opaquePsoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
	opaquePsoDesc.DSVFormat = mDepthStencilFormat;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&opaquePsoDesc, IID_PPV_ARGS(&mPSOs["opaque"])));


    //
    // PSO for opaque wireframe objects.
    //

    D3D12_GRAPHICS_PIPELINE_STATE_DESC opaqueWireframePsoDesc = opaquePsoDesc;
    opaqueWireframePsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&opaqueWireframePsoDesc, IID_PPV_ARGS(&mPSOs["opaque_wireframe"])));
}

void ShapesApp::BuildFrameResources()
{
    for(int i = 0; i < gNumFrameResources; ++i)
    {
        mFrameResources.push_back(std::make_unique<FrameResource>(md3dDevice.Get(),
            1, (UINT)mAllRitems.size()));
    }
}

void ShapesApp::BuildRenderItems()
{
    auto gridRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&gridRitem->World, XMMatrixScaling(100.0f, 100.0f, 10.0f) * XMMatrixTranslation(100.0f, -100.0f, -13.0f));
    gridRitem->World = MathHelper::Identity4x4();
    gridRitem->ObjCBIndex = 0;
    gridRitem->Geo = mGeometries["shapeGeo"].get();
    gridRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    gridRitem->IndexCount = gridRitem->Geo->DrawArgs["grid"].IndexCount;
    gridRitem->StartIndexLocation = gridRitem->Geo->DrawArgs["grid"].StartIndexLocation;
    gridRitem->BaseVertexLocation = gridRitem->Geo->DrawArgs["grid"].BaseVertexLocation;
    mAllRitems.push_back(std::move(gridRitem));

    auto leftWallRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&leftWallRitem->World, XMMatrixScaling(3.0f, 19.5f, 30.0f) * XMMatrixTranslation(-15.0f, 8.0f, 0.0f));
    leftWallRitem->ObjCBIndex = 1;
    leftWallRitem->Geo = mGeometries["shapeGeo"].get();
    leftWallRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    leftWallRitem->IndexCount = leftWallRitem->Geo->DrawArgs["wall"].IndexCount;
    leftWallRitem->StartIndexLocation = leftWallRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    leftWallRitem->BaseVertexLocation = leftWallRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(leftWallRitem));

   

    auto rightWallRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&rightWallRitem->World, XMMatrixScaling(3.0f, 19.5f, 30.0f) * XMMatrixTranslation(+15.0f, 8.0f, 0.0f));
    rightWallRitem->ObjCBIndex = 2;
    rightWallRitem->Geo = mGeometries["shapeGeo"].get();
    rightWallRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    rightWallRitem->IndexCount = rightWallRitem->Geo->DrawArgs["wall"].IndexCount;
    rightWallRitem->StartIndexLocation = rightWallRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    rightWallRitem->BaseVertexLocation = rightWallRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(rightWallRitem));

    auto backWallRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&backWallRitem->World, XMMatrixScaling(30.0f, 19.5f, 3.0f) * XMMatrixTranslation(0.0f, 8.0f, 15.0f));
    backWallRitem->ObjCBIndex = 3;
    backWallRitem->Geo = mGeometries["shapeGeo"].get();
    backWallRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    backWallRitem->IndexCount = backWallRitem->Geo->DrawArgs["wall"].IndexCount;
    backWallRitem->StartIndexLocation = backWallRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    backWallRitem->BaseVertexLocation = backWallRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(backWallRitem));

    auto gateLeftRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&gateLeftRitem->World, XMMatrixScaling(10.0f, 19.5f, 3.0f) * XMMatrixTranslation(-10.0f, 8.0f, -15.0f));
    gateLeftRitem->ObjCBIndex = 4;
    gateLeftRitem->Geo = mGeometries["shapeGeo"].get();
    gateLeftRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    gateLeftRitem->IndexCount = gateLeftRitem->Geo->DrawArgs["wall"].IndexCount;
    gateLeftRitem->StartIndexLocation = gateLeftRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    gateLeftRitem->BaseVertexLocation = gateLeftRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(gateLeftRitem));

    auto gateRightRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&gateRightRitem->World, XMMatrixScaling(10.0f, 19.5f, 3.0f) * XMMatrixTranslation(10.0f, 8.0f, -15.0f));
    gateRightRitem->ObjCBIndex = 5;
    gateRightRitem->Geo = mGeometries["shapeGeo"].get();
    gateRightRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    gateRightRitem->IndexCount = gateRightRitem->Geo->DrawArgs["wall"].IndexCount;
    gateRightRitem->StartIndexLocation = gateRightRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    gateRightRitem->BaseVertexLocation = gateRightRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(gateRightRitem));

    auto gateTopRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&gateTopRitem->World, XMMatrixScaling(10.0f, 5.0f, 3.0f) * XMMatrixTranslation(0.0f, 15.25f, -15.0f));
    gateTopRitem->ObjCBIndex = 6;
    gateTopRitem->Geo = mGeometries["shapeGeo"].get();
    gateTopRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    gateTopRitem->IndexCount = gateTopRitem->Geo->DrawArgs["wall"].IndexCount;
    gateTopRitem->StartIndexLocation = gateTopRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    gateTopRitem->BaseVertexLocation = gateTopRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(gateTopRitem));

    auto blTowerRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&blTowerRitem->World, XMMatrixScaling(4.0f, 25.0f, 4.0f) * XMMatrixTranslation(-15.0f, 8.0f, 15.0f));
    blTowerRitem->ObjCBIndex = 7;
    blTowerRitem->Geo = mGeometries["shapeGeo"].get();
    blTowerRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    blTowerRitem->IndexCount = blTowerRitem->Geo->DrawArgs["tower"].IndexCount;
    blTowerRitem->StartIndexLocation = blTowerRitem->Geo->DrawArgs["tower"].StartIndexLocation;
    blTowerRitem->BaseVertexLocation = blTowerRitem->Geo->DrawArgs["tower"].BaseVertexLocation;
    mAllRitems.push_back(std::move(blTowerRitem));

    auto brTowerRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&brTowerRitem->World, XMMatrixScaling(4.0f, 25.0f, 4.0f) * XMMatrixTranslation(+15.0f, 8.0f, 15.0f));
    brTowerRitem->ObjCBIndex = 8;
    brTowerRitem->Geo = mGeometries["shapeGeo"].get();
    brTowerRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    brTowerRitem->IndexCount = brTowerRitem->Geo->DrawArgs["tower"].IndexCount;
    brTowerRitem->StartIndexLocation = brTowerRitem->Geo->DrawArgs["tower"].StartIndexLocation;
    brTowerRitem->BaseVertexLocation = brTowerRitem->Geo->DrawArgs["tower"].BaseVertexLocation;
    mAllRitems.push_back(std::move(brTowerRitem));

    auto flTowerRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&flTowerRitem->World, XMMatrixScaling(4.0f, 25.0f, 4.0f) * XMMatrixTranslation(-15.0f, 8.0f, -15.0f));
    flTowerRitem->ObjCBIndex = 9;
    flTowerRitem->Geo = mGeometries["shapeGeo"].get();
    flTowerRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    flTowerRitem->IndexCount = flTowerRitem->Geo->DrawArgs["tower"].IndexCount;
    flTowerRitem->StartIndexLocation = flTowerRitem->Geo->DrawArgs["tower"].StartIndexLocation;
    flTowerRitem->BaseVertexLocation = flTowerRitem->Geo->DrawArgs["tower"].BaseVertexLocation;
    mAllRitems.push_back(std::move(flTowerRitem));

    auto frTowerRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&frTowerRitem->World, XMMatrixScaling(4.0f, 25.0f, 4.0f)* XMMatrixTranslation(+15.0f, 8.0f, -15.0f));
    frTowerRitem->ObjCBIndex = 10;
    frTowerRitem->Geo = mGeometries["shapeGeo"].get();
    frTowerRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    frTowerRitem->IndexCount = frTowerRitem->Geo->DrawArgs["tower"].IndexCount;
    frTowerRitem->StartIndexLocation = frTowerRitem->Geo->DrawArgs["tower"].StartIndexLocation;
    frTowerRitem->BaseVertexLocation = frTowerRitem->Geo->DrawArgs["tower"].BaseVertexLocation;
    mAllRitems.push_back(std::move(frTowerRitem));

    auto blTopTowerRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&blTopTowerRitem->World, XMMatrixScaling(8.0f, 3.0f, 8.0f)* XMMatrixTranslation(-15.0f, 21.0f, 15.0f));
    blTopTowerRitem->ObjCBIndex = 11;
    blTopTowerRitem->Geo = mGeometries["shapeGeo"].get();
    blTopTowerRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    blTopTowerRitem->IndexCount = blTopTowerRitem->Geo->DrawArgs["wall"].IndexCount;
    blTopTowerRitem->StartIndexLocation = blTopTowerRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    blTopTowerRitem->BaseVertexLocation = blTopTowerRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(blTopTowerRitem));

    auto brTopTowerRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&brTopTowerRitem->World, XMMatrixScaling(8.0f, 3.0f, 8.0f)* XMMatrixTranslation(15.0f, 21.0f, 15.0f));
    brTopTowerRitem->ObjCBIndex = 12;
    brTopTowerRitem->Geo = mGeometries["shapeGeo"].get();
    brTopTowerRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    brTopTowerRitem->IndexCount = brTopTowerRitem->Geo->DrawArgs["wall"].IndexCount;
    brTopTowerRitem->StartIndexLocation = brTopTowerRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    brTopTowerRitem->BaseVertexLocation = brTopTowerRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(brTopTowerRitem));

    auto flTopTowerRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&flTopTowerRitem->World, XMMatrixScaling(8.0f, 3.0f, 8.0f)* XMMatrixTranslation(-15.0f, 21.0f, -15.0f));
    flTopTowerRitem->ObjCBIndex = 13;
    flTopTowerRitem->Geo = mGeometries["shapeGeo"].get();
    flTopTowerRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    flTopTowerRitem->IndexCount = flTopTowerRitem->Geo->DrawArgs["wall"].IndexCount;
    flTopTowerRitem->StartIndexLocation = flTopTowerRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    flTopTowerRitem->BaseVertexLocation = flTopTowerRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(flTopTowerRitem));

    auto frTopTowerRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&frTopTowerRitem->World, XMMatrixScaling(8.0f, 3.0f, 8.0f)* XMMatrixTranslation(15.0f, 21.0f, -15.0f));
    frTopTowerRitem->ObjCBIndex = 14;
    frTopTowerRitem->Geo = mGeometries["shapeGeo"].get();
    frTopTowerRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    frTopTowerRitem->IndexCount = frTopTowerRitem->Geo->DrawArgs["wall"].IndexCount;
    frTopTowerRitem->StartIndexLocation = frTopTowerRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    frTopTowerRitem->BaseVertexLocation = frTopTowerRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(frTopTowerRitem));

    auto blTowerConeRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&blTowerConeRitem->World, XMMatrixScaling(3.5f, 14.0f, 3.5f)* XMMatrixTranslation(-15.0f, 28.0f, 15.0f));
    blTowerConeRitem->ObjCBIndex = 15;
    blTowerConeRitem->Geo = mGeometries["shapeGeo"].get();
    blTowerConeRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    blTowerConeRitem->IndexCount = blTowerConeRitem->Geo->DrawArgs["cone"].IndexCount;
    blTowerConeRitem->StartIndexLocation = blTowerConeRitem->Geo->DrawArgs["cone"].StartIndexLocation;
    blTowerConeRitem->BaseVertexLocation = blTowerConeRitem->Geo->DrawArgs["cone"].BaseVertexLocation;
    mAllRitems.push_back(std::move(blTowerConeRitem));

    auto brTowerConeRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&brTowerConeRitem->World, XMMatrixScaling(3.5f, 14.0f, 3.5f)* XMMatrixTranslation(15.0f, 28.0f, 15.0f));
    brTowerConeRitem->ObjCBIndex = 16;
    brTowerConeRitem->Geo = mGeometries["shapeGeo"].get();
    brTowerConeRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    brTowerConeRitem->IndexCount = brTowerConeRitem->Geo->DrawArgs["cone"].IndexCount;
    brTowerConeRitem->StartIndexLocation = brTowerConeRitem->Geo->DrawArgs["cone"].StartIndexLocation;
    brTowerConeRitem->BaseVertexLocation = brTowerConeRitem->Geo->DrawArgs["cone"].BaseVertexLocation;
    mAllRitems.push_back(std::move(brTowerConeRitem));

    auto flTowerConeRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&flTowerConeRitem->World, XMMatrixScaling(3.5f, 14.0f, 3.5f)* XMMatrixTranslation(-15.0f, 28.0f, -15.0f));
    flTowerConeRitem->ObjCBIndex = 17;
    flTowerConeRitem->Geo = mGeometries["shapeGeo"].get();
    flTowerConeRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    flTowerConeRitem->IndexCount = flTowerConeRitem->Geo->DrawArgs["cone"].IndexCount;
    flTowerConeRitem->StartIndexLocation = flTowerConeRitem->Geo->DrawArgs["cone"].StartIndexLocation;
    flTowerConeRitem->BaseVertexLocation = flTowerConeRitem->Geo->DrawArgs["cone"].BaseVertexLocation;
    mAllRitems.push_back(std::move(flTowerConeRitem));

    auto frTowerConeRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&frTowerConeRitem->World, XMMatrixScaling(3.5f, 14.0f, 3.5f)* XMMatrixTranslation(15.0f, 28.0f, -15.0f));
    frTowerConeRitem->ObjCBIndex = 18;
    frTowerConeRitem->Geo = mGeometries["shapeGeo"].get();
    frTowerConeRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    frTowerConeRitem->IndexCount = frTowerConeRitem->Geo->DrawArgs["cone"].IndexCount;
    frTowerConeRitem->StartIndexLocation = frTowerConeRitem->Geo->DrawArgs["cone"].StartIndexLocation;
    frTowerConeRitem->BaseVertexLocation = frTowerConeRitem->Geo->DrawArgs["cone"].BaseVertexLocation;
    mAllRitems.push_back(std::move(frTowerConeRitem));

    auto floorRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&floorRitem->World, XMMatrixScaling(30.0f, 3.5f, 30.0f)* XMMatrixTranslation(0.0f, 1.5f, 0.0f));
    floorRitem->ObjCBIndex = 19;
    floorRitem->Geo = mGeometries["shapeGeo"].get();
    floorRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    floorRitem->IndexCount = floorRitem->Geo->DrawArgs["floor"].IndexCount;
    floorRitem->StartIndexLocation = floorRitem->Geo->DrawArgs["floor"].StartIndexLocation;
    floorRitem->BaseVertexLocation = floorRitem->Geo->DrawArgs["floor"].BaseVertexLocation;
    mAllRitems.push_back(std::move(floorRitem));

    auto wedgeRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&wedgeRitem->World, XMMatrixScaling(10.0f, 3.7f, 18.0f)* XMMatrixTranslation(0.0f, 1.5f, -24.0f));
    wedgeRitem->ObjCBIndex = 20;
    wedgeRitem->Geo = mGeometries["shapeGeo"].get();
    wedgeRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    wedgeRitem->IndexCount = wedgeRitem->Geo->DrawArgs["wedge"].IndexCount;
    wedgeRitem->StartIndexLocation = wedgeRitem->Geo->DrawArgs["wedge"].StartIndexLocation;
    wedgeRitem->BaseVertexLocation = wedgeRitem->Geo->DrawArgs["wedge"].BaseVertexLocation;
    mAllRitems.push_back(std::move(wedgeRitem));

    auto powerPyramidRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&powerPyramidRitem->World, XMMatrixScaling(8.0f, 7.5f, 8.0f)* XMMatrixTranslation(0.0f, 6.5f, 0.0f));
    powerPyramidRitem->ObjCBIndex = 21;
    powerPyramidRitem->Geo = mGeometries["shapeGeo"].get();
    powerPyramidRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    powerPyramidRitem->IndexCount = powerPyramidRitem->Geo->DrawArgs["pyramid"].IndexCount;
    powerPyramidRitem->StartIndexLocation = powerPyramidRitem->Geo->DrawArgs["pyramid"].StartIndexLocation;
    powerPyramidRitem->BaseVertexLocation = powerPyramidRitem->Geo->DrawArgs["pyramid"].BaseVertexLocation;
    mAllRitems.push_back(std::move(powerPyramidRitem));

    auto poewrGemRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&poewrGemRitem->World, XMMatrixScaling(2.0f, 3.0f, 2.0f)* XMMatrixTranslation(0.0f, 15.0f, 0.0f));
    poewrGemRitem->ObjCBIndex = 22;
    poewrGemRitem->Geo = mGeometries["shapeGeo"].get();
    poewrGemRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    poewrGemRitem->IndexCount = poewrGemRitem->Geo->DrawArgs["diamond"].IndexCount;
    poewrGemRitem->StartIndexLocation = poewrGemRitem->Geo->DrawArgs["diamond"].StartIndexLocation;
    poewrGemRitem->BaseVertexLocation = poewrGemRitem->Geo->DrawArgs["diamond"].BaseVertexLocation;
    mAllRitems.push_back(std::move(poewrGemRitem));

   

	

    UINT objCBIndex = 23;
    for (int i = 0; i < 3; ++i)
    {
        auto leftMerlin = std::make_unique<RenderItem>();
        auto rightMerlin = std::make_unique<RenderItem>();
        

        XMMATRIX leftMerlinWorld = XMMatrixScaling(3.0f, 3.0f, 3.0f) * XMMatrixTranslation(-15.0f, 19.0f, -6.0f + i * 6.0f);
        XMMATRIX rightMerlinWorld = XMMatrixScaling(3.0f, 3.0f, 3.0f) * XMMatrixTranslation(+15.0f, 19.0f, -6.0f + i * 6.0f);
        

        XMStoreFloat4x4(&leftMerlin->World, rightMerlinWorld);
        leftMerlin->ObjCBIndex = objCBIndex++;
        leftMerlin->Geo = mGeometries["shapeGeo"].get();
        leftMerlin->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        leftMerlin->IndexCount = leftMerlin->Geo->DrawArgs["wall"].IndexCount;
        leftMerlin->StartIndexLocation = leftMerlin->Geo->DrawArgs["wall"].StartIndexLocation;
        leftMerlin->BaseVertexLocation = leftMerlin->Geo->DrawArgs["wall"].BaseVertexLocation;

        XMStoreFloat4x4(&rightMerlin->World, leftMerlinWorld);
        rightMerlin->ObjCBIndex = objCBIndex++;
        rightMerlin->Geo = mGeometries["shapeGeo"].get();
        rightMerlin->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        rightMerlin->IndexCount = rightMerlin->Geo->DrawArgs["wall"].IndexCount;
        rightMerlin->StartIndexLocation = rightMerlin->Geo->DrawArgs["wall"].StartIndexLocation;
        rightMerlin->BaseVertexLocation = rightMerlin->Geo->DrawArgs["wall"].BaseVertexLocation;



        mAllRitems.push_back(std::move(leftMerlin));
        mAllRitems.push_back(std::move(rightMerlin));

    }

    auto innerLeftWallRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&innerLeftWallRitem->World, XMMatrixScaling(1.0f, 19.5f, 13.0f) * XMMatrixTranslation(-6.0f, 8.0f, 0.0f));
    //XMStoreFloat4x4(&leftWallRitem->World, XMMatrixScaling(3.0f, 19.5f, 30.0f) * XMMatrixTranslation(-15.0f, 8.0f, 0.0f));
    innerLeftWallRitem->ObjCBIndex = 24;
    innerLeftWallRitem->Geo = mGeometries["shapeGeo"].get();
    innerLeftWallRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    innerLeftWallRitem->IndexCount = innerLeftWallRitem->Geo->DrawArgs["wall"].IndexCount;
    innerLeftWallRitem->StartIndexLocation = innerLeftWallRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    innerLeftWallRitem->BaseVertexLocation = innerLeftWallRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(innerLeftWallRitem));

   auto innerrightWallRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&innerrightWallRitem->World, XMMatrixScaling(1.0f, 19.5f, 13.0f) * XMMatrixTranslation(+6.0f, 8.0f, 0.0f));
    innerrightWallRitem->ObjCBIndex = 25;
    innerrightWallRitem->Geo = mGeometries["shapeGeo"].get();
    innerrightWallRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    innerrightWallRitem->IndexCount = innerrightWallRitem->Geo->DrawArgs["wall"].IndexCount;
    innerrightWallRitem->StartIndexLocation = innerrightWallRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    innerrightWallRitem->BaseVertexLocation = innerrightWallRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(innerrightWallRitem));

    auto innerbackWallRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&innerbackWallRitem->World, XMMatrixScaling(12.0f, 19.5f, 1.0f) * XMMatrixTranslation(0.0f, 8.0f, 6.0f));
    innerbackWallRitem->ObjCBIndex = 26;
    innerbackWallRitem->Geo = mGeometries["shapeGeo"].get();
    innerbackWallRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    innerbackWallRitem->IndexCount = innerbackWallRitem->Geo->DrawArgs["wall"].IndexCount;
    innerbackWallRitem->StartIndexLocation = innerbackWallRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    innerbackWallRitem->BaseVertexLocation = innerbackWallRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(innerbackWallRitem));

    auto innergateLeftRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&innergateLeftRitem->World, XMMatrixScaling(4.0f, 19.5f, 1.0f) * XMMatrixTranslation(-4.0f, 8.0f, -6.0f));
    innergateLeftRitem->ObjCBIndex = 27;
    innergateLeftRitem->Geo = mGeometries["shapeGeo"].get();
    innergateLeftRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    innergateLeftRitem->IndexCount = innergateLeftRitem->Geo->DrawArgs["wall"].IndexCount;
    innergateLeftRitem->StartIndexLocation = innergateLeftRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    innergateLeftRitem->BaseVertexLocation = innergateLeftRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(innergateLeftRitem));

    auto innergateRightRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&innergateRightRitem->World, XMMatrixScaling(4.0f, 19.5f, 1.0f) * XMMatrixTranslation(4.0f, 8.0f, -6.0f));
    innergateRightRitem->ObjCBIndex = 28;
    innergateRightRitem->Geo = mGeometries["shapeGeo"].get();
    innergateRightRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    innergateRightRitem->IndexCount = innergateRightRitem->Geo->DrawArgs["wall"].IndexCount;
    innergateRightRitem->StartIndexLocation = innergateRightRitem->Geo->DrawArgs["wall"].StartIndexLocation;
    innergateRightRitem->BaseVertexLocation = innergateRightRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(innergateRightRitem));

    auto topLeftPlatformitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&topLeftPlatformitem->World, XMMatrixScaling(9.0f, 1.0f, 6.0f) * XMMatrixTranslation(-10.0f, 18.0f, 0.0f));
    topLeftPlatformitem->ObjCBIndex = 29;
    topLeftPlatformitem->Geo = mGeometries["shapeGeo"].get();
    topLeftPlatformitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    topLeftPlatformitem->IndexCount = topLeftPlatformitem->Geo->DrawArgs["wall"].IndexCount;
    topLeftPlatformitem->StartIndexLocation = topLeftPlatformitem->Geo->DrawArgs["wall"].StartIndexLocation;
    topLeftPlatformitem->BaseVertexLocation = topLeftPlatformitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(topLeftPlatformitem));

    auto topRightPlatformitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&topRightPlatformitem->World, XMMatrixScaling(9.0f, 1.0f, 6.0f) * XMMatrixTranslation(9.0f, 18.0f, 0.0f));
    topRightPlatformitem->ObjCBIndex = 30;
    topRightPlatformitem->Geo = mGeometries["shapeGeo"].get();
    topRightPlatformitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    topRightPlatformitem->IndexCount = topRightPlatformitem->Geo->DrawArgs["wall"].IndexCount;
    topRightPlatformitem->StartIndexLocation = topRightPlatformitem->Geo->DrawArgs["wall"].StartIndexLocation;
    topRightPlatformitem->BaseVertexLocation = topRightPlatformitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(topRightPlatformitem));

    auto topBackPlatformitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&topBackPlatformitem->World, XMMatrixScaling(6.0f, 1.0f, 9.0f)* XMMatrixTranslation(0.0f, 18.0f, 10.0f));
    topBackPlatformitem->ObjCBIndex = 31;
    topBackPlatformitem->Geo = mGeometries["shapeGeo"].get();
    topBackPlatformitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    topBackPlatformitem->IndexCount = topBackPlatformitem->Geo->DrawArgs["wall"].IndexCount;
    topBackPlatformitem->StartIndexLocation = topBackPlatformitem->Geo->DrawArgs["wall"].StartIndexLocation;
    topBackPlatformitem->BaseVertexLocation = topBackPlatformitem->Geo->DrawArgs["wall"].BaseVertexLocation;
    mAllRitems.push_back(std::move(topBackPlatformitem));

	// All the render items are opaque.
	for(auto& e : mAllRitems)
		mOpaqueRitems.push_back(e.get());
}


//The DrawRenderItems method is invoked in the main Draw call:
void ShapesApp::DrawRenderItems(ID3D12GraphicsCommandList* cmdList, const std::vector<RenderItem*>& ritems)
{
    UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));
 
	auto objectCB = mCurrFrameResource->ObjectCB->Resource();

    // For each render item...
    for(size_t i = 0; i < ritems.size(); ++i)
    {
        auto ri = ritems[i];

        cmdList->IASetVertexBuffers(0, 1, &ri->Geo->VertexBufferView());
        cmdList->IASetIndexBuffer(&ri->Geo->IndexBufferView());
        cmdList->IASetPrimitiveTopology(ri->PrimitiveType);

        // Offset to the CBV in the descriptor heap for this object and for this frame resource.
        UINT cbvIndex = mCurrFrameResourceIndex*(UINT)mOpaqueRitems.size() + ri->ObjCBIndex;
        auto cbvHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(mCbvHeap->GetGPUDescriptorHandleForHeapStart());
        cbvHandle.Offset(cbvIndex, mCbvSrvUavDescriptorSize);

        cmdList->SetGraphicsRootDescriptorTable(0, cbvHandle);

        cmdList->DrawIndexedInstanced(ri->IndexCount, 1, ri->StartIndexLocation, ri->BaseVertexLocation, 0);
    }
}


