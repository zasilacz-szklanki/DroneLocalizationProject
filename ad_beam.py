#porównanie skuteczności i wydajności lokalizacji drona przy użyciu algorytmów: DAS, Ortogonalny, Clean-SC, CMF.;

OPENBLAS_NUM_THREADS=1

import sys
import time
from datetime import datetime
from pathlib import Path
import acoular as ac
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.io import wavfile

class TeeOutput:
    def __init__(self, stream, filename):
        self.file = open(filename, "w")
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __del__(self):
        self.file.close()

sys.stdout = TeeOutput(sys.stdout,"logs.log")
sys.stderr = TeeOutput(sys.stderr,"errs.log")

i = 0

def run(algorithm, inputfile_path):

    ac.config.global_caching = 'none'

    # Parametry
    sfreq = 96000
    duration = 1
    # nsamples = duration * sfreq
    
    micgeofile = Path('ring8_capstone_wall.xml')
    inputfile = Path(inputfile_path)
    
    
    samplerate, wavData = wavfile.read(inputfile)
    ts = ac.TimeSamples(data=wavData, sample_freq=samplerate)
    
    mg = ac.MicGeom(file=micgeofile)
    rg = ac.RectGrid(
        x_min=-2,
        x_max=+2,
        y_min=-2,
        y_max=+2,
        z=1,
        increment=0.1,
    )
    st = ac.SteeringVector(grid=rg, mics=mg)
    
    frg_span = 0.2
    FPS = 30
    frames_count = int(ts.num_samples / ts.sample_freq * FPS)
    frame_length = int(ts.sample_freq / FPS)
    print("Klatki do wygenerowania: ", frames_count)
    
    gen = ts.result(frame_length)
    
    fig, ax = plt.subplots()

    def init():
        ax.clear()
        ax.axis("off")

    # Funkcja do animacji
    def update(frame):
        global i
        i += 1
        print(f"\rKlatka {i}/{frames_count}", end="", flush=True, file=sys.stderr)
        res = frame[0]
        p = frame[1]
        fres = frame[2]
        fp = frame[3]
        ax.clear()
        ax.imshow(
            np.transpose(res),
            extent=rg.extend(),
            origin="lower",
        )
        ax.imshow(
            np.transpose(fres),
            extent=(p[0] - frg_span, p[0] + frg_span, p[1] - frg_span, p[1] + frg_span),
            origin="lower",
        )
        ax.plot(fp[0], fp[1], 'r+')
        ax.annotate(f'({fp[0]:0,.2f}, {fp[1]:0,.2f})',
                    xy=(fp[0], fp[1]),
                    xytext=(fp[0] + frg_span, fp[1] + frg_span),
                    color='white')
        
    def mapIndexToRange(i, num, v_min=0, v_max=1):
        step = (v_max - v_min) / (num - 1)
        return v_min + (i * step)

    avg_frame_time = 0;
    max_frame_time = 0;
    min_frame_time = 0;
    t1 = time.thread_time()
    pt = 0
    frames = list()

    for block in gen:
        pt1 = time.thread_time()
        perf_counter_start=time.perf_counter()
        global i
        tempData = block
        tempTS = ac.TimeSamples(data=tempData, sample_freq=samplerate)
        ps = ac.PowerSpectra(source=tempTS, block_size=128, overlap='50%', window='Hanning')

        # Wybór algorytmu beamformingowego
        if algorithm == 'BeamformerBase':
            bb = ac.BeamformerBase(freq_data=ps, steer=st)
        elif algorithm == 'BeamformerOrth':
            bb = ac.BeamformerOrth(freq_data=ps, steer=st)
        elif algorithm == 'BeamformerCleansc':
            bb = ac.BeamformerCleansc(freq_data=ps, steer=st)
        elif algorithm == 'BeamformerCMF':
            bb = ac.BeamformerCMF(freq_data=ps, steer=st)
        else:
            raise ValueError(f"Nieznany algorytm: {algorithm}")

        tempRes = np.sum(bb.result[4:32], 0)
        r = tempRes.reshape(rg.shape)
        p = np.unravel_index(np.argmax(r), r.shape)
        px = mapIndexToRange(p[0], r.shape[0], rg.extend()[0], rg.extend()[1])
        py = mapIndexToRange(p[1], r.shape[1], rg.extend()[2], rg.extend()[3])
        
        pt2 = time.thread_time()
        pt += pt2 - pt1
        
        frg = ac.RectGrid(
            x_min = px - frg_span,
            x_max = px + frg_span,
            y_min = py - frg_span,
            y_max = py + frg_span,
            z=1,
            increment=0.01,
        )
        fst = ac.SteeringVector(grid=frg, mics=mg, steer_type='classic')

        # Ponowne użycie tego samego algorytmu w drugim etapie
        if algorithm == 'BeamformerBase':
            fbb = ac.BeamformerBase(freq_data=ps, steer=fst)
        elif algorithm == 'BeamformerOrth':
            fbb = ac.BeamformerOrth(freq_data=ps, steer=fst)
        elif algorithm == 'BeamformerCleansc':
            fbb = ac.BeamformerCleansc(freq_data=ps, steer=fst)
        elif algorithm == 'BeamformerCMF':
            fbb = ac.BeamformerCMF(freq_data=ps, steer=fst)
        else:
            raise ValueError(f"Nieznany algorytm: {algorithm}")

        tempFRes = np.sum(fbb.result[8:16], 0)
        fr = tempFRes.reshape(frg.shape)
        fp = np.unravel_index(np.argmax(fr), fr.shape)
        fpx = mapIndexToRange(fp[0], fr.shape[0], frg.extend()[0], frg.extend()[1])
        fpy = mapIndexToRange(fp[1], fr.shape[1], frg.extend()[2], frg.extend()[3])
        
        frames.append((r, (px, py), fr, (fpx, fpy)))
        perf_counter_stop=time.perf_counter()-perf_counter_start
        avg_frame_time += perf_counter_stop
        max_frame_time = max(max_frame_time,perf_counter_stop)
        if(i==0):
            min_frame_time=perf_counter_stop
        min_frame_time = min(min_frame_time,perf_counter_stop)
        print(f"\rWywołanie bf: {i} \tCzas: {perf_counter_stop}", end="", flush=True, file=sys.stderr)
        i += 1
    
    print()
    
    t2 = time.thread_time()

    print(f"Całko. czas dla ramek:\t{avg_frame_time}");
    print(f"Średni czas dla ramki:\t{avg_frame_time/i}");
    print(f"Max. czas dla ramki:\t{max_frame_time}");
    print(f"Min. czas dla ramki:\t{min_frame_time}");
    i = 0

    print(f"Czas dla 1 etapu bf:\t{pt}")
    print(f"Czas dla 2 etapu bf:\t{t2-t1}")
    
    points = np.array([ p[1] for p in frames ])
    focus_points = np.array([ p[3] for p in frames ])

    np.save(f"./output/{inputfile.stem}_{algorithm}_points.npy", points)
    print(f"Zapisano: ./output/{inputfile.stem}_{algorithm}_points.npy")
    np.save(f"./output/{inputfile.stem}_{algorithm}_focuspoints.npy", focus_points)
    print(f"Zapisano: ./output/{inputfile.stem}_{algorithm}_focuspoints.npy")

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, repeat=True, interval=1 / FPS)
    ani.save(f"./output/{inputfile.stem}_{algorithm}_map.mp4", writer="ffmpeg", fps=FPS)
    print(f"\nZapisano: ./output/{inputfile.stem}_{algorithm}_map.mp4")
    i = 0#
    plt.close()

if __name__ == '__main__':
    algorithms = ['BeamformerBase']#, 'BeamformerOrth']#, 'BeamformerCleansc', 'BeamformerCMF']
    recordings = [
        #"./audio/wall1.wav"#,
        #"./audio/wall2.wav",
        "./audio/wall3.wav"
    ]

    for rec in recordings:
        for algo in algorithms:
            print(f"\nCzas uruchomienia: {datetime.now()}")
            print(f"Przetwarzanie {rec} algorytmem {algo}")
            run(algorithm=algo, inputfile_path=rec)
