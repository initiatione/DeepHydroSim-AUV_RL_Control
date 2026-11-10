#include "MyAUVApp.h"
#include "MyAUVManager.h"
#include <cfenv>
#include <cstdlib>
#include <iostream>

int main(int argc, const char *argv[])
{
    // feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    // feenableexcept(FE_INVALID | FE_OVERFLOW);

    sf::RenderSettings s;
    s.windowW = 1200;
    s.windowH = 900;
    s.aa = sf::RenderQuality::HIGH;
    s.shadows = sf::RenderQuality::HIGH;
    s.ao = sf::RenderQuality::HIGH;
    s.atmosphere = sf::RenderQuality::MEDIUM;
    s.ocean = sf::RenderQuality::HIGH;
    s.ssr = sf::RenderQuality::HIGH;

    sf::HelperSettings h;
    h.showFluidDynamics = false;
    h.showCoordSys = true;
    h.showBulletDebugInfo = false;
    h.showSensors = false;
    h.showActuators = false;
    h.showForces = true;

    MyAUVManager simulationManager(200.0);
    // Wall-clock speedup only: the physics timestep is unchanged, so the
    // logged data is identical, just produced faster. Override with
    // SF_REALTIME_FACTOR=10 to collect hours of data in minutes.
    double realtimeFactor = 1.0;
    if (const char* env = std::getenv("SF_REALTIME_FACTOR")) {
        realtimeFactor = std::atof(env);
        if (realtimeFactor <= 0.0) realtimeFactor = 1.0;
    }
    std::cout << "[Main] Realtime factor: " << realtimeFactor << std::endl;
    simulationManager.setRealtimeFactor(realtimeFactor);
    MyAUVApp app(std::string(DATA_DIR_PATH), s, h, &simulationManager);
    app.Run();

    return 0;
}
