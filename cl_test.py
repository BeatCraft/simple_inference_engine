import pyopencl as cl

platforms = cl.get_platforms()

for platform in platforms:
    print(f"Platform Name: {platform.name}")
    print(f"Platform Vendor: {platform.vendor}")
    
    devices = platform.get_devices()
    
    for device in devices:
        print(f"\tDevice Name: {device.name}")
        print(f"\tDevice Type: {cl.device_type.to_string(device.type)}")
        print(f"\tDevice Vendor: {device.vendor}")
        print(f"\tDevice Version: {device.version}\n")
    #
#


