#ifndef __SCENE_LOADER_H__
#define __SCENE_LOADER_H__

#include <vector>

// New, modernized API: fills output vectors with scene data.
// position, velocity and color are vectors of length 3 * numCircles.
// radius is length numCircles.
void loadCircleScene(
    SceneName sceneName,
    int &numCircles,
    std::vector<float> &position,
    std::vector<float> &velocity,
    std::vector<float> &color,
    std::vector<float> &radius);

#endif
