# Header
std_msgs/Header header

# String to give the sequence ID
string sequence_id

# Azimuth angles in radians
float32[] azimuth

# Timestamps in seconds
int64[] timestamps

# Azimuth resolution in radians
float64 resolution

# Radar chirp data (flattened)
uint8[] chirps
