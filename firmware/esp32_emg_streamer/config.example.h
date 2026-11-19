// Copy this file to config.h (same folder) and fill in your values.
// config.h is gitignored so your WiFi password never ends up on GitHub.
#pragma once

#define WIFI_SSID     "your-wifi-name"
#define WIFI_PASSWORD "your-wifi-password"

// IP of the PC (or Quest) running `myobridge listen` / `myobridge forward`.
// Find it with `ipconfig` (Windows) or `ip addr` / `ifconfig` (Linux/macOS).
#define HOST_IP   "192.168.1.50"
#define HOST_PORT 5005
