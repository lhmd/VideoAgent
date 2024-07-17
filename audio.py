import pyaudio
import requests

# dependency
# ipcam app: https://itunes.apple.com/app/id440270152

# install
# sudo apt install portaudio19-dev
# pip install requests, pyaudio

url = 'http://192.168.50.218/audio.pcm'
boundary = b'--myboundary'
# Parameters
channels = 1  # Assuming mono
sample_rate = 8000  # Sample rate as specified
sample_width = 2  # Assuming 16-bit audio (2 bytes per sample)
chunk_size = 4096 # larger so no boundary is split in the middle, should not be too large tho
# Initialize PyAudio
p = pyaudio.PyAudio()
# Open stream
stream = p.open(format=p.get_format_from_width(sample_width),
                channels=channels,
                rate=sample_rate,
                output=True)
with requests.get(url, stream=True) as r:
    data_buffer = b''
    for chunk in r.iter_content(chunk_size=chunk_size):
        # Check for boundary and process accordingly
        if boundary in chunk:
            # Split the chunk by boundary
            parts = chunk.split(boundary)
            for part in parts:
                # if b'Content-Type: audio/x-wav' in part:
                    # Skip header lines and get actual PCM data
                header_end = part.find(b'\r\n\r\n') + 4
                pcm_data = part[header_end:]
                stream.write(pcm_data)
                #elif len(part) > 0:
                #    # Write remaining PCM data directly
                #    stream.write(part)
        else:
            # If no boundary, just write the chunk
            # print(chunk)
            stream.write(chunk)

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()