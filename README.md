# python
from tkinter import *
import tkinter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading  # For running the heart rate monitoring in parallel
from scipy.signal import butter, filtfilt

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(1, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Heart rate monitoring function
def monitor_heart_rate():
    plt.switch_backend('Agg')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    fps = 30
    buffer_size = fps * 10  # 10 seconds buffer
    roi_intensity_signal = []  # Store mean intensity signal
    heart_rate_signal = []  # Store heart rate signal for filtering

    # Face detection initialization
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize plot for ROI intensity signal
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)

    # Set y-axis range from 50 to 150
    ax.set_ylim(50, 150)
    ax.set_xlim(0, buffer_size)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean ROI Intensity')
    ax.set_title('Mean Intensity Signal (Face Region)')

    def update_plot(signal):
        if len(signal) == 0:
            return
        line.set_xdata(np.arange(len(signal)))
        line.set_ydata(signal)
        ax.set_xlim(0, max(len(signal), buffer_size))
        fig.canvas.draw()
        fig.canvas.flush_events()

    def save_plot_as_image():
        plt.savefig('roi_intensity_signal_plot.png')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi = frame[y:y + h, x:x + w]

                # Compute mean intensity for the green channel of the ROI
                mean_green_intensity = np.mean(roi[:, :, 1])  # Using the green channel
                roi_intensity_signal.append(mean_green_intensity)
                heart_rate_signal.append(mean_green_intensity)

                # Limit the size of the signal arrays
                if len(roi_intensity_signal) > buffer_size:
                    roi_intensity_signal = roi_intensity_signal[-buffer_size:]
                    heart_rate_signal = heart_rate_signal[-buffer_size:]

                # Basic heart rate estimation
                estimated_basic_heart_rate = int(mean_green_intensity)

                # Check if we have enough data points before filtering
                if len(heart_rate_signal) > 33:  # Ensure enough points for the filter
                    fs = fps
                    lowcut = 0.75
                    highcut = 4.0
                    filtered_signal = bandpass_filter(heart_rate_signal, lowcut, highcut, fs)

                    # Estimate heart rate from the filtered signal
                    if len(filtered_signal) > 0:
                        heart_rate_estimation = np.mean(filtered_signal)
                        estimated_heart_rate = int(heart_rate_estimation)

                # Display the basic heart rate estimation
                cv2.putText(frame, f"Remote Heart Rate is : {estimated_basic_heart_rate}", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                update_plot(roi_intensity_signal)
                save_plot_as_image()

                # Load the saved plot and display it next to the video frame
                plot_img = cv2.imread('roi_intensity_signal_plot.png')
                plot_img = cv2.resize(plot_img, (frame.shape[1], frame.shape[0]))  # Resize plot to match video frame
                combined_frame = np.hstack((frame, plot_img))  # Combine video and plot side by side
                cv2.imshow("Webcam with Mean ROI Intensity Signal", combined_frame)
        else:
            # Show the frame without the plot if no face is detected
            cv2.imshow("Webcam with Mean ROI Intensity Signal", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Function to handle starting the monitoring in a separate thread
def startMonitoring():
    threading.Thread(target=monitor_heart_rate).start()

# GUI Code
main = tkinter.Tk()
main.title("Heart Rate Estimation using Green-Verkruysse Method")
main.geometry("500x400")

# Exit function
def exit_app():
    main.destroy()

# GUI Layout
font = ('times', 16, 'bold')
title = Label(main, text='Heart Rate Estimation Green-Verkruysse Method', anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
start_button = Button(main, text="Start Heart Rate Monitoring Using Webcam", command=startMonitoring)
start_button.place(x=50, y=200)
start_button.config(font=font1)

exit_button = Button(main, text="Exit", command=exit_app)
exit_button.place(x=50, y=300)
exit_button.config(font=font1)

main.config(bg='blue1')
main.mainloop()

