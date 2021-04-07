from __future__ import print_function, division
import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os
from mediaplayer import *


class VideoRecorder():
    "Video class based on openCV"
    def __init__(self, camId, name="./recordings/temp_video.mp4", fourcc="MJPG", sizex=640, sizey=480, fps=30):
        self.open = True
        self.device_index = camId    # 0 or camindex  web camera
        self.fps = fps
        self.fourcc = fourcc
        self.frameSize = (sizex, sizey)
        self.video_filename = name
        self.video_cap = cv2.VideoCapture(self.device_index)
        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1
        self.start_time = time.time()

    def record(self):
        timer_start = time.time()
        timer_current = 0
        while self.open:
            ret, video_frame = self.video_cap.read()
            if ret:
                self.video_out.write(video_frame)
                self.frame_counts += 1
                time.sleep(1/self.fps)
                # gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

                cv2.imshow('video_frame',video_frame)
                cv2.waitKey(1)
            else:
                break
        cv2.destroyAllWindows()

    def stop(self):
        "Finishes the video recording therefore the thread too"
        if self.open:
            # self.open=False
            self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()

    def start(self):
        "Launches the video recording function using a thread"

        video_thread = threading.Thread(target=self.record)
        video_thread.start()

class AudioRecorder():
    "Audio class based on pyAudio and Wave"
    def __init__(self, filename="./recordings/temp_audio.wav", rate=44100, fpb=1024, channels=2):
        self.open = True
        self.rate = rate
        self.frames_per_buffer = fpb
        self.channels = channels
        self.format = pyaudio.paInt16
        self.audio_filename = filename
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                          channels=self.channels,
                                          rate=self.rate,
                                          input=True,
                                          frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []

    def record(self):
        "Audio starts being recorded"
        self.stream.start_stream()
        while self.open:
            data = self.stream.read(self.frames_per_buffer)
            self.audio_frames.append(data)
            if not self.open:
                break

    def stop(self):
        "Finishes the audio recording therefore the thread too"
        if self.open:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()


    def start(self):
        "Launches the audio recording function using a thread"
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()

class start_AVrecording():
    def __init__(self,camId,filename):

        global video_thread
        global audio_thread
        self.cameraDeviceId = camId
        self.filename = filename
        self.audio_thread = AudioRecorder()
        self.video_thread = VideoRecorder(camId)

    def start(self,camId,filename):
        self.video_thread = VideoRecorder(camId)
        self.audio_thread = AudioRecorder()
        self.audio_thread.start()
        self.video_thread.start()
        return filename

    def start_video_recording(self):
        global video_thread
        self.video_thread.start()
        return self.filename


    def start_AVrecording(self):
        global video_thread
        global audio_thread
        self.audio_thread.start()
        self.video_thread.start()
        return self.filename


    def start_audio_recording(self):
        global audio_thread
        self.audio_thread.start()
        return self.filename

    def stop(self):
        self.audio_thread.stop()
        frame_counts = self.video_thread.frame_counts
        elapsed_time = time.time() - self.video_thread.start_time
        recorded_fps = frame_counts / elapsed_time
        # print("total frames " + str(frame_counts))
        # print("elapsed time " + str(elapsed_time))
        # print("recorded fps " + str(recorded_fps))

        self.video_thread.stop()
        local_path = os.getcwd()

        # Makes sure the threads have finished
        # while threading.active_count() > 1:
        #     time.sleep(1)
        # Merging audio and video signal
        Executable = f'{os.getcwd()}' + r'\ffmpeg\bin\ffmpeg.exe'

        if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected
            print("Re-encoding")
            print(recorded_fps)
            remove_file(str(local_path)+"/recordings/temp_video2.mp4")
            cmd = Executable + ' -r ' + str(recorded_fps) + ' -i ./recordings/temp_video.mp4 -pix_fmt yuv420p -r 6 ./recordings/temp_video2.mp4'
            # cmd = 'ffmpeg -r ' + str(recorded_fps) + ' -i ./recordings/temp_video.mp4 -pix_fmt yuv420p -r 6 ./recordings/temp_video2.mp4'
            subprocess.call(cmd)
            print("Muxing")
            cmd = Executable + ' -y -ac 2 -channel_layout stereo -i ./recordings/temp_audio.wav -i ./recordings/temp_video2.mp4 -pix_fmt yuv420p  ' + self.filename + '.mp4'
            # cmd = 'ffmpeg -y -ac 2 -channel_layout stereo -i ./recordings/temp_audio.wav -i ./recordings/temp_video2.mp4 -pix_fmt yuv420p ' + self.filename + '.mp4'
            subprocess.call(cmd)
            # subprocess.call(Executable + 'kill -e -q ffmpeg')




        else:
            print("Normal recording\nMuxing")
            cmd = Executable + '-y -ac 2 -channel_layout stereo -i ./recordings/temp_audio.wav -i temp_video.mp4 -pix_fmt yuv420p ' + self.filename + '.mp4'
            # cmd = 'ffmpeg -y -ac 2 -channel_layout stereo -i ./recordings/temp_audio.wav -i temp_video.mp4 -pix_fmt yuv420p ' + self.filename + '.mp4'
            subprocess.call(cmd)
            print("..")



def remove_file(filename):
    if os.path.exists(str(filename)):
        os.remove(str(filename))

#
def file_manager(filename="test"):
    "Required and wanted processing of final files"
    local_path = os.getcwd()
    remove_file(str(local_path)+"/recordings/temp_audio.wav")
    remove_file(str(local_path)+"/recordings/temp_video.avi")
    remove_file(str(local_path)+"/recordings/temp_video.mp4")
    remove_file(str(local_path)+"/recordings/temp_video2.mp4")
    # if os.path.exists(str(local_path) + "/recordings/temp_video2.avi"):
    #     os.remove(str(local_path) + "/recordings/temp_video2.avi")
    # if os.path.exists(str(local_path) + "/recordings/temp_video2.mp4"):
    #     os.remove(str(local_path) + "/recordings/temp_video2.mp4")
    # if os.path.exists(str(local_path) + "/" + filename + ".avi"):
    #     os.remove(str(local_path) + "/" + filename + ".avi")

if __name__ == '__main__':

    a = start_AVrecording(camId = 0, filename="test")
    a.start(camId=1, filename="test")

    print("press 'q' to stop recording")
    if input()=="q":
        a.stop()
