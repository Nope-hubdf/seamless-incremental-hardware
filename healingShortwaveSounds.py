from myhdl import block, Signal, intbv, always_comb, always_seq, delay
import numpy as np

# Constants
SAMPLE_RATE = 44100  # Hz
LUT_SIZE = 1024  # Size of sine lookup table
BIT_WIDTH = 16  # Fixed-point bit width
MAX_VAL = 2**(BIT_WIDTH-1) - 1  # Max value for 16-bit signed
SINE_LUT = [int(np.sin(2 * np.pi * i / LUT_SIZE) * MAX_VAL) for i in range(LUT_SIZE)]

@block
def sine_oscillator(clk, freq, amplitude, output, sample_rate=SAMPLE_RATE, lut_size=LUT_SIZE):
    """Sine oscillator with phase accumulator and LUT."""
    phase = Signal(intbv(0)[32:])  # 32-bit phase accumulator
    phase_inc = Signal(intbv(0)[32:])  # Phase increment per sample
    lut_index = Signal(intbv(0)[10:])  # Index into LUT (log2(LUT_SIZE)=10)

    @always_comb
    def calc_phase_inc():
        # Phase increment = (freq * LUT_SIZE) / SAMPLE_RATE
        phase_inc.next = int((freq * lut_size) // sample_rate)

    @always_seq(clk.posedge, reset=None)
    def phase_acc():
        # Update phase accumulator
        phase.next = (phase + phase_inc) % (lut_size << 22)  # Wrap at LUT_SIZE * 2^22
        lut_index.next = phase[32:22]  # Top 10 bits for LUT index

    @always_comb
    def output_logic():
        # Scale sine output by amplitude
        sine_val = SINE_LUT[lut_index]
        output.next = (sine_val * amplitude) // MAX_VAL

    return calc_phase_inc, phase_acc, output_logic

@block
def patch1(clk, output):
    """Patch 1: Mix.fill(8, {SinOsc.ar(rrand(20,200),0,1/8/4)})"""
    outputs = [Signal(intbv(0, min=-MAX_VAL, max=MAX_VAL+1)) for _ in range(8)]
    amplitudes = [Signal(intbv(int(MAX_VAL / 8 / 4))) for _ in range(8)]  # 1/8/4 scaling
    freqs = [Signal(intbv(f)) for f in [20, 40, 60, 80, 100, 120, 160, 200]]  # Random freqs

    oscs = [sine_oscillator(clk, freqs[i], amplitudes[i], outputs[i]) for i in range(8)]

    @always_comb
    def mix():
        # Sum outputs, ensure no overflow
        total = sum(o for o in outputs)
        output.next = total // 8  # Scale down to prevent clipping

    return oscs, mix

@block
def patch2(clk, output):
    """Patch 2: Mix.fill(8, {SinOsc.ar(rrand(20,200)*SinOsc.ar([1,2,4,8].choose),0,1/8/4)})"""
    outputs = [Signal(intbv(0, min=-MAX_VAL, max=MAX_VAL+1)) for _ in range(8)]
    amplitudes = [Signal(intbv(int(MAX_VAL / 8 / 4))) for _ in range(8)]
    base_freqs = [Signal(intbv(f)) for f in [20, 40, 60, 80, 100, 120, 160, 200]]
    mod_freqs = [Signal(intbv(f)) for f in [1, 2, 4, 8, 1, 2, 4, 8]]  # [1,2,4,8].choose
    mod_outputs = [Signal(intbv(0, min=-MAX_VAL, max=MAX_VAL+1)) for _ in range(8)]
    mod_amplitudes = [Signal(intbv(MAX_VAL)) for _ in range(8)]  # Full amplitude for modulator

    mod_oscs = [sine_oscillator(clk, mod_freqs[i], mod_amplitudes[i], mod_outputs[i]) for i in range(8)]
    oscs = [None] * 8

    @always_comb
    def freq_mod():
        for i in range(8):
            # Frequency modulation: base_freq * (1 + mod_output/MAX_VAL)
            mod_val = (mod_outputs[i] >> 8) + MAX_VAL // 256  # Scale mod output
            oscs[i] = sine_oscillator(clk, (base_freqs[i] * mod_val) // (MAX_VAL // 256), amplitudes[i], outputs[i])

    @always_comb
    def mix():
        total = sum(o for o in outputs)
        output.next = total // 8

    return mod_oscs, freq_mod, oscs, mix

@block
def patch3(clk, output):
    """Patch 3: Mix.fill(8, {SinOsc.ar(rrand(20,200)*SinOsc.ar([1,2,4,8].choose),0,SinOsc.ar([1,2,4,8].choose*0.01,0,1/8/4,0.01))})"""
    outputs = [Signal(intbv(0, min=-MAX_VAL, max=MAX_VAL+1)) for _ in range(8)]
    base_freqs = [Signal(intbv(f)) for f in [20, 40, 60, 80, 100, 120, 160, 200]]
    mod_freqs = [Signal(intbv(f)) for f in [1, 2, 4, 8, 1, 2, 4, 8]]
    amp_mod_freqs = [Signal(intbv(int(f * 0.01))) for f in [1, 2, 4, 8, 1, 2, 4, 8]]
    mod_outputs = [Signal(intbv(0, min=-MAX_VAL, max=MAX_VAL+1)) for _ in range(8)]
    amp_mod_outputs = [Signal(intbv(0, min=-MAX_VAL, max=MAX_VAL+1)) for _ in range(8)]
    amplitudes = [Signal(intbv(0, min=0, max=MAX_VAL+1)) for _ in range(8)]
    base_amplitude = int(MAX_VAL / 8 / 4)

    mod_oscs = [sine_oscillator(clk, mod_freqs[i], Signal(intbv(MAX_VAL)), mod_outputs[i]) for i in range(8)]
    amp_mod_oscs = [sine_oscillator(clk, amp_mod_freqs[i], Signal(intbv(MAX_VAL)), amp_mod_outputs[i]) for i in range(8)]
    oscs = [None] * 8

    @always_comb
    def mod_logic():
        for i in range(8):
            # Frequency modulation
            mod_val = (mod_outputs[i] >> 8) + MAX_VAL // 256
            freq = (base_freqs[i] * mod_val) // (MAX_VAL // 256)
            # Amplitude modulation: (amp_mod_output * 0.01 + 0.01) * base_amplitude
            amp_val = ((amp_mod_outputs[i] >> 8) + (MAX_VAL // 100)) * base_amplitude // MAX_VAL
            amplitudes[i].next = amp_val
            oscs[i] = sine_oscillator(clk, freq, amplitudes[i], outputs[i])

    @always_comb
    def mix():
        total = sum(o for o in outputs)
        output.next = total // 8

    return mod_oscs, amp_mod_oscs, mod_logic, oscs, mix

@block
def patch4(clk, output):
    """Patch 4: Mix.fill(8, {SinOsc.ar(rrand(20,200),0,SinOsc.ar([1,2,4,8].choose*0.01,0,1/8/4,0.01))})"""
    outputs = [Signal(intbv(0, min=-MAX_VAL, max=MAX_VAL+1)) for _ in range(8)]
    freqs = [Signal(intbv(f)) for f in [20, 40, 60, 80, 100, 120, 160, 200]]
    amp_mod_freqs = [Signal(intbv(int(f * 0.01))) for f in [1, 2, 4, 8, 1, 2, 4, 8]]
    amp_mod_outputs = [Signal(intbv(0, min=-MAX_VAL, max=MAX_VAL+1)) for _ in range(8)]
    amplitudes = [Signal(intbv(0, min=0, max=MAX_VAL+1)) for _ in range(8)]
    base_amplitude = int(MAX_VAL / 8 / 4)

    amp_mod_oscs = [sine_oscillator(clk, amp_mod_freqs[i], Signal(intbv(MAX_VAL)), amp_mod_outputs[i]) for i in range(8)]
    oscs = [sine_oscillator(clk, freqs[i], amplitudes[i], outputs[i]) for i in range(8)]

    @always_comb
    def amp_mod():
        for i in range(8):
            amp_val = ((amp_mod_outputs[i] >> 8) + (MAX_VAL // 100)) * base_amplitude // MAX_VAL
            amplitudes[i].next = amp_val

    @always_comb
    def mix():
        total = sum(o for o in outputs)
        output.next = total // 8

    return amp_mod_oscs, oscs, amp_mod, mix

@block
def testbench():
    """Testbench to simulate patch1 and collect output samples."""
    clk = Signal(bool(0))
    output = Signal(intbv(0, min=-MAX_VAL, max=MAX_VAL+1))
    samples = []

    # Instantiate patch1
    dut = patch1(clk, output)

    @always(delay(1000000 // SAMPLE_RATE))  # Clock period for 44.1 kHz
    def clkgen():
        clk.next = not clk

    @always(clk.posedge)
    def collect():
        samples.append(int(output))

    return dut, clkgen, collect

# Simulate and collect samples
def simulate_patch1():
    tb = testbench()
    tb.simulate(sim_cycles=44100)  # Simulate for 1 second
    return tb.samples

if __name__ == "__main__":
    # Run simulation for patch1
    samples = simulate_patch1()
    # Optionally save to WAV file for audio playback
    from scipy.io.wavfile import write
    write("patch1_output.wav", SAMPLE_RATE, np.array(samples, dtype=np.int16))