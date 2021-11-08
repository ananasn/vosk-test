import queue

import sounddevice as sd
import vosk

q = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    q.put(bytes(indata))


model = 'model'
device = 0
device_info = sd.query_devices(0, 'input')
samplerate = int(device_info['default_samplerate'])

try:
    model = vosk.Model(model)

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000,
                           device=device, dtype='int16',
                           channels=1, callback=callback):

        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                print(rec.Result())
            else:
                print(rec.PartialResult())

except KeyboardInterrupt:
    print('\nDone')
except Exception as e:
    print(type(e).__name__ + ': ' + str(e))
