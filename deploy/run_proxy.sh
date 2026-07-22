#!/bin/bash
cd /opt/aspida-src
exec /opt/aspida-src/obj/openai_proxy 127.0.0.1 8765 "$(cat /opt/aspida-src/server_pub.hex)" 8099
