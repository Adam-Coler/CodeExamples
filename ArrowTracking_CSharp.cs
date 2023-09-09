// The arrow points users to where they need to look prior to the video stimuli playing
// this causes all participants to start viewing the stimuli at the same location

void Update() // every frame while this object active
{
    // make a new 3-tuple the represents where we want the arrow to be
    // that location is placed in front of the headset as far away as the target the arrow points towards is
    Vector3 lerpTarget = Camera.main.transform.position + Camera.main.transform.forward * Vector3.Distance(Camera.main.transform.position, target.transform.position);

    // the alpha of the color for the arrow is set based on its distance from the target
    var distance = Vector3.Distance(indicator.transform.position, target.transform.position);
    Color color = indicatorImage.color;
    // that arrow alpha is 1 when the arrow is far from the target then one the arrow is within .15 meters it starts to become transparent
    color.a = Mathf.Min(distance - .5f, 1f);

    // set the color
    indicatorImage.color = color;

    // set the position of the arrow. Linear interpolation is used to smooth movement between frames, the delta time is the time elapsed since the last frame, this way the rate of movement is constant
    indicator.transform.position = Vector3.Lerp(indicator.transform.position, lerpTarget, Time.deltaTime);
    // now we set the arrow to face the users head
    indicator.transform.LookAt(target.transform.position);
}
