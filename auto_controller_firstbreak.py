import firstbreak
import json
import picoscopereader as psdata
import plug_parameter as plug
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    data = open('measure_information_automatic_firstbreak.json', 'r')
    data_json = json.load(data)
    experiment_information = data_json['experiment_information']

    for experiment in experiment_information:
        face_to_face_file = experiment["face_to_face_file"]
        sample_information = experiment["sample_information"]
        
        time, waves = psdata.load_psdata_bufferized(face_to_face_file, False, "buffer")
        t0 = firstbreak.first_break2(time, waves[0])

        label = 'Face to face time = {:.2f} us'.format(t0)

        plt.figure()
        plt.suptitle(face_to_face_file)
        plt.subplot(1, 1, 1)
        plt.plot(time, waves[0], "C7")
        plt.vlines(t0, np.nanmin(waves[0]), np.nanmax(waves[0]), colors='C0', label=label)
        plt.xlim(0.0, np.nanmax(time))
        plt.legend()

        print("Face to face file: {}\nFirst break: {:.2f} us\n".format(face_to_face_file, t0))

        for sample in sample_information:
            sample_file = sample["file"]
            length = sample["length"]
            sample_type = sample["type"]
            time_factor = sample["time_convertion_factor"]
        
            time, waves = psdata.load_psdata_bufferized(sample_file, False, "buffer")
            time *= time_factor
            velocities = plug.standard_velocities(sample_type, 'material_velocities.csv')
            vmin = velocities[0]/1000.0
            vmax = velocities[1]/1000.0

            t_first_break = firstbreak.first_break2(time, waves[0], t0, length=length, vmin=vmin, vmax=vmax, run_twice=True)

            V = length/t_first_break
            V *= 1000.0
            label = 'First break time = {:.2f} us\nVelocity = {:.2f} m/s'.format(t_first_break + t0, V)

            plt.figure()
            plt.suptitle(sample_file)
            plt.subplot(1, 1, 1)
            plt.plot(time, waves[0], "C7")
            plt.vlines(t_first_break, np.nanmin(waves[0]), np.nanmax(waves[0]), colors='C0', label=label)
            plt.xlim(0.0, np.nanmax(time))
            plt.legend()

            print("Sample file: {}\nFirst break: {:.2f} us\nVelocity: {:.2f} m/s\n".format(sample_file, t_first_break + t0, V))
        print("")
    
    plt.show()

