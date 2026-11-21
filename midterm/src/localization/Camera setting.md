
def Camera "wrist_Cam_02" (
        kind = "component"
    )
    {
        float2 clippingRange = (0.01, 10000000)
        float focalLength = 1.88
        float focusDistance = 20
        float horizontalAperture = 3.11
        custom bool omni:kit:cameraLock = 1
        float verticalAperture = 2.33
        }
    ![[Pasted image 20260430133813.png]]

def Camera "front_Cam" (
        kind = "component"
    )
    {
        float2 clippingRange = (0.01, 10000000)
        float focalLength = 1
        float focusDistance = 20
        float horizontalAperture = 2.04
        custom bool omni:kit:cameraLock = 1
        float verticalAperture = 1.53
        }
    ![[Pasted image 20260430133836.png]]

def Camera "head_Cam" (
        kind = "component"
    )
    {
        float2 clippingRange = (0.01, 10000000)
        float focalLength = 1
        float focusDistance = 20
        float horizontalAperture = 2.04
        custom bool omni:kit:cameraLock = 1
        float verticalAperture = 1.53
        }
    ![[Pasted image 20260430133900.png]]