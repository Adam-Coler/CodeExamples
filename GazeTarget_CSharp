// This code tracked whether people were looking at the gaze target that would trigger the video playing
// and displayed where they were looking by using a in game object


void Update()    // called each frame
{
    // Get gaze data if gaze is allowed and calibrated
    if (VarjoEyeTracking.IsGazeAllowed() && VarjoEyeTracking.IsGazeCalibrated())
    {
        // Show the gaze target
        gazeTarget.SetActive(showGazePoint);

        // and show the fixation point (where the user is looking)
        fixationPoint.SetActive(showFixationPoint);

        // get the gaze data from the varjo api
        gazeData = VarjoEyeTracking.GetGaze();

        // if gaze is being tracked
        if (gazeData.status != VarjoEyeTracking.GazeStatus.Invalid)
        {
            // GazeRay vectors are relative to the HMD pose so they need to be transformed to world space
            // Set gaze origin as raycast origin
            rayOrigin = xrCamera.transform.TransformPoint(gazeData.gaze.origin);

            // Set gaze direction as raycast direction
            direction = xrCamera.transform.TransformDirection(gazeData.gaze.forward);

            // Fixation point can be calculated using ray origin, direction and focus distance
            fixationPointTransform.position = rayOrigin + direction * gazeData.focusDistance;
        }
    }

    // as a fallback use the active camera representing where the user is looking if no device is detected
    if (!XRSettings.isDeviceActive)
    {
        rayOrigin = Camera.main.transform.position;
        direction = Camera.main.transform.forward;
    }

        // Raycast to world from VR Camera position towards fixation point
        if (Physics.SphereCast(rayOrigin, gazeRadius, direction, out hit))
    {
        // Put target on gaze raycast position with an offset towards user
        // the gaze target is instantiated first then generates this script
        // it sets itself as the gazeTarget object
        gazeTarget.transform.position = hit.point - direction * targetOffset;

        // Make the gaze target point towards user
        gazeTarget.transform.LookAt(rayOrigin, Vector3.up);

        // Scale gazetarget with distance so it appears to be always same size
        distance = hit.distance;
        gazeTarget.transform.localScale = Vector3.one * distance;

        // if object hit by the ray cast is tagged as the goal target start the countdown
        if (hit.transform.CompareTag("GazeTargetEnter"))
        {
            // start the countdown by calling the script attached to the object
            hit.transform.parent.GetComponent<GazeTargetCountdown>().startCountdown();
        } else if (hit.transform.CompareTag("GazeTargetExit"))
        {
            // if the hit object is near the target but not in the center it is tagged as leaving the intended target
            // then stop the countdown
            hit.transform.parent.GetComponent<GazeTargetCountdown>().stopCountdown();
        }

    }
    else
    {
        // If gaze ray didn't hit anything, the gaze target is shown at fixed distance
        gazeTarget.transform.position = rayOrigin + direction * floatingGazeTargetDistance;
        // turn the target towards the users eyes maintaining it's vertical axis
        gazeTarget.transform.LookAt(rayOrigin, Vector3.up);
        // and scale it such that it appears to be the same size, in VR the user can move by moving their body in the real world
        // this controls for that movement
        gazeTarget.transform.localScale = Vector3.one * floatingGazeTargetDistance;
    }


}

