import numpy as np
import netsquid as ns
from netsquid.nodes.node import Node
from netsquid.components.qprocessor import QuantumProcessor,PhysicalInstruction
from netsquid.components.instructions import INSTR_X,INSTR_H,INSTR_MEASURE,INSTR_MEASURE_X
from netsquid.components.models.qerrormodels import FibreLossModel,T1T2NoiseModel,DepolarNoiseModel,DephaseNoiseModel
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.models import  FibreDelayModel

from difflib import SequenceMatcher
from random import Random

import alice
import bob

from utils import ManualFibreLossModel
from cascade import cascade_reconcile
from privacy_amplification import toeplitz_hash

import logging
logger = logging.getLogger(__name__)


def _setup_nodes_and_channels(
    num_bits,
    fibreLen,
    memNoiseMmodel,
    processorNoiseModel,
    qdelay,
    sourceFreq,
    qSpeed,
    cSpeed,
    fibreNoise,
):
    nodeA = Node("Alice", port_names=["portQA_1", "portCA_1", "portCA_2"])
    nodeB = Node("Bob", port_names=["portQB_1", "portCB_1", "portCB_2"])

    Alice_processor = QuantumProcessor(
        "processor_A",
        num_positions=2 * 10**2,
        mem_noise_models=memNoiseMmodel,
        phys_instructions=[
            PhysicalInstruction(INSTR_X, duration=5, quantum_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_H, duration=5, quantum_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_MEASURE, duration=3700, quantum_noise_model=processorNoiseModel, parallel=True),
            PhysicalInstruction(INSTR_MEASURE_X, duration=3700, quantum_noise_model=processorNoiseModel, parallel=True),
        ],
    )

    Bob_processor = QuantumProcessor(
        "processor_B",
        num_positions=2 * 10**2,
        mem_noise_models=memNoiseMmodel,
        phys_instructions=[
            PhysicalInstruction(INSTR_X, duration=5, quantum_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_H, duration=5, quantum_noise_model=processorNoiseModel),
            PhysicalInstruction(INSTR_MEASURE, duration=3700, quantum_noise_model=processorNoiseModel, parallel=True),
            PhysicalInstruction(INSTR_MEASURE_X, duration=3700, quantum_noise_model=processorNoiseModel, parallel=True),
        ],
    )

    MyQChannel = QuantumChannel(
        "QChannel_A->B",
        delay=qdelay,
        length=fibreLen,
        models={
            "myFibreLossModel": FibreLossModel(p_loss_init=0, p_loss_length=0, rng=None),
            "mydelay_model": FibreDelayModel(c=qSpeed),
            "myFibreNoiseModel": DepolarNoiseModel(depolar_rate=fibreNoise, time_independent=False),
        },
    )

    nodeA.connect_to(
        nodeB,
        MyQChannel,
        local_port_name=nodeA.ports["portQA_1"].name,
        remote_port_name=nodeB.ports["portQB_1"].name,
    )

    MyCChannel = ClassicalChannel(
        "CChannel_B->A", delay=0, length=fibreLen, models={"myCDelayModel": FibreDelayModel(c=cSpeed)}
    )
    MyCChannel2 = ClassicalChannel(
        "CChannel_A->B", delay=0, length=fibreLen, models={"myCDelayModel": FibreDelayModel(c=cSpeed)}
    )

    nodeB.connect_to(nodeA, MyCChannel, local_port_name="portCB_1", remote_port_name="portCA_1")
    nodeA.connect_to(nodeB, MyCChannel2, local_port_name="portCA_2", remote_port_name="portCB_2")

    Alice_protocol = alice.Protocol(nodeA, Alice_processor, num_bits, sourceFreq=sourceFreq)
    Bob_protocol = bob.Protocol(nodeB, Bob_processor, num_bits)

    return Alice_protocol, Bob_protocol


def _run_bb84_once(
    num_bits,
    fibreLen,
    memNoiseMmodel,
    processorNoiseModel,
    qdelay,
    sourceFreq,
    qSpeed,
    cSpeed,
    fibreNoise,
    run_label,
):
    ns.sim_reset()

    Alice_protocol, Bob_protocol = _setup_nodes_and_channels(
        num_bits,
        fibreLen,
        memNoiseMmodel,
        processorNoiseModel,
        qdelay,
        sourceFreq,
        qSpeed,
        cSpeed,
        fibreNoise,
    )

    Bob_protocol.start()
    Alice_protocol.start()

    startTime = ns.util.simtools.sim_time(magnitude=ns.NANOSECOND)
    logger.info("%s run start: num_bits=%s fibreLen=%s", run_label, num_bits, fibreLen)
    ns.sim_run()

    endTime = Bob_protocol.endTime
    logger.info("%s run end: sim_time_ns=%s", run_label, endTime - startTime)

    return Alice_protocol.key, Bob_protocol.key, startTime, endTime

def run_BB84_sim(
    runtimes=1,
    num_bits=20,
    fibreLen=10**-9,
    memNoiseMmodel=None,
    processorNoiseModel=None,
    loss_init=0,
    loss_len=0,
    qdelay=0,
    sourceFreq=8e7,
    lenLoss=0,
    qSpeed=2 * 10**5,
    cSpeed=2 * 10**5,
    fibreNoise=0,
    cascade_block_size=16,
    cascade_rounds=4,
    cascade_final_single_bit_pass=False,
    pa_output_frac=0.5,
    pa_seed=0,
):
    MyKeyList_A=[]  # local protocol list A
    MyKeyList_B=[]  # local protocol list B
    MyKeyList_B_corrected=[]
    MyKeyList_A_final=[]
    MyKeyList_B_final=[]
    MyKeyRateList=[]
    MyCascadeStats=[]
    
    for _ in range(runtimes):
        firstKey, secondKey, startTime, endTime = _run_bb84_once(
            num_bits,
            fibreLen,
            memNoiseMmodel,
            processorNoiseModel,
            qdelay,
            sourceFreq,
            qSpeed,
            cSpeed,
            fibreNoise,
            "BB84_baseline",
        )

        logger.info(
            "Sifting complete (no loss): key_len_A=%s key_len_B=%s",
            len(firstKey),
            len(secondKey),
        )
        logger.debug("Alice key (sifted):%s", firstKey)
        logger.debug("Bob key (sifted):%s", secondKey)

        qber = _bit_error_rate(firstKey, secondKey)
        logger.info("Baseline QBER:%s", qber)

        corrected_b_key, stats = cascade_reconcile(
            firstKey,
            secondKey,
            block_size=cascade_block_size,
            rounds=cascade_rounds,
            seed=None,
            final_single_bit_pass=cascade_final_single_bit_pass,
        )
        logger.info(
            "Cascade complete: corrections=%s parity_checks=%s",
            stats["corrections"],
            stats["parity_checks"],
        )

        pa_len = int(len(firstKey) * pa_output_frac)
        final_a_key = toeplitz_hash(firstKey, pa_len, seed=pa_seed)
        final_b_key = toeplitz_hash(corrected_b_key, pa_len, seed=pa_seed)
        logger.info("Privacy amplification: output_len=%s", pa_len)

        MyKeyList_A.append(firstKey)
        MyKeyList_B.append(secondKey)
        MyKeyList_B_corrected.append(corrected_b_key)
        MyKeyList_A_final.append(final_a_key)
        MyKeyList_B_final.append(final_b_key)
        MyCascadeStats.append(stats)

        s = SequenceMatcher(None, final_a_key, final_b_key)
        MyKeyRateList.append(len(final_b_key) * s.ratio() / (endTime - startTime) * 10**9)

    return (
        MyKeyList_A,
        MyKeyList_B,
        MyKeyList_B_corrected,
        MyKeyList_A_final,
        MyKeyList_B_final,
        MyKeyRateList,
        MyCascadeStats,
    )


def _bit_error_rate(key_a, key_b):
    n = min(len(key_a), len(key_b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if key_a[i] != key_b[i]) / n


def _apply_bit_flip_noise(key, bit_flip_prob, rng):
    if bit_flip_prob <= 0:
        return list(key)
    noisy = []
    for bit in key:
        if rng.random() < bit_flip_prob:
            noisy.append(bit ^ 1)
        else:
            noisy.append(bit)
    return noisy


def run_BB84_sim_with_noise(
    runtimes=1,
    num_bits=20,
    fibreLen=10**-9,
    memNoiseMmodel=None,
    processorNoiseModel=None,
    loss_init=0.02,
    loss_len=0.005,
    qdelay=0,
    sourceFreq=8e7,
    lenLoss=0,
    qSpeed=2 * 10**5,
    cSpeed=2 * 10**5,
    fibreNoise=0,
    bit_flip_prob=0.08,
    cascade_block_size=16,
    cascade_rounds=4,
    cascade_final_single_bit_pass=False,
    pa_output_frac=0.5,
    pa_seed=0,
    seed=None,
):
    """
    BB84 simulation that injects classical bit-flip noise and applies Cascade.
    """
    MyKeyList_A = []
    MyKeyList_B_noisy = []
    MyKeyList_B_corrected = []
    MyKeyList_A_final = []
    MyKeyList_B_final = []
    MyKeyRateList = []
    MyCascadeStats = []

    rng = Random(seed)

    for _ in range(runtimes):
        firstKey, secondKey, startTime, endTime = _run_bb84_once(
            num_bits,
            fibreLen,
            memNoiseMmodel,
            processorNoiseModel,
            qdelay,
            sourceFreq,
            qSpeed,
            cSpeed,
            fibreNoise,
            "BB84+noise",
        )

        # apply loss to simulate random hardware issues or eavesdropping
        effective_init_loss = loss_init
        effective_len_loss = loss_len if loss_len else lenLoss
        firstKey, secondKey = ManualFibreLossModel(
            key1=firstKey,
            key2=secondKey,
            numNodes=2,
            fibreLen=fibreLen,
            iniLoss=effective_init_loss,
            lenLoss=effective_len_loss,
            algorithmFator=2,
        )
        logger.info(
            "After simulated loss: key_len_A=%s key_len_B=%s (init=%s len=%s)",
            len(firstKey),
            len(secondKey),
            effective_init_loss,
            effective_len_loss,
        )
        logger.debug("Alice key (post-loss):%s", firstKey)
        logger.debug("Bob key (post-loss):%s", secondKey)

        noisy_b_key = _apply_bit_flip_noise(secondKey, bit_flip_prob, rng)
        ber_before = _bit_error_rate(firstKey, noisy_b_key)
        logger.info(
            "Noise injected: bit_flip_prob=%s ber_before=%s",
            bit_flip_prob,
            ber_before,
        )

        corrected_b_key, stats = cascade_reconcile(
            firstKey,
            noisy_b_key,
            block_size=cascade_block_size,
            rounds=cascade_rounds,
            seed=seed,
            final_single_bit_pass=cascade_final_single_bit_pass,
        )
        ber_after = _bit_error_rate(firstKey, corrected_b_key)
        logger.info(
            "Cascade complete: ber_after=%s corrections=%s parity_checks=%s",
            ber_after,
            stats["corrections"],
            stats["parity_checks"],
        )

        pa_len = int(len(firstKey) * pa_output_frac)
        final_a_key = toeplitz_hash(firstKey, pa_len, seed=pa_seed)
        final_b_key = toeplitz_hash(corrected_b_key, pa_len, seed=pa_seed)
        logger.info("Privacy amplification: output_len=%s", pa_len)

        MyKeyList_A.append(firstKey)
        MyKeyList_B_noisy.append(noisy_b_key)
        MyKeyList_B_corrected.append(corrected_b_key)
        MyKeyList_A_final.append(final_a_key)
        MyKeyList_B_final.append(final_b_key)
        MyCascadeStats.append(stats)

        s = SequenceMatcher(None, final_a_key, final_b_key)
        MyKeyRateList.append(len(final_b_key) * s.ratio() / (endTime - startTime) * 10**9)

    return (
        MyKeyList_A,
        MyKeyList_B_noisy,
        MyKeyList_B_corrected,
        MyKeyList_A_final,
        MyKeyList_B_final,
        MyKeyRateList,
        MyCascadeStats,
    )


def run_BB84_sim_with_noise_overflow(
    runtimes=1,
    num_bits=20,
    fibreLen=10**-9,
    memNoiseMmodel=None,
    processorNoiseModel=None,
    loss_init=0.2,
    loss_len=0.05,
    qdelay=0,
    sourceFreq=8e7,
    lenLoss=0,
    qSpeed=2 * 10**5,
    cSpeed=2 * 10**5,
    fibreNoise=0,
    bit_flip_prob=0.02,
    cascade_block_size=16,
    cascade_rounds=4,
    cascade_final_single_bit_pass=False,
    pa_output_frac=0.5,
    pa_seed=0,
    qber_reject_threshold=0.05,
    seed=None,
):
    """
    BB84 simulation with high noise; aborts if QBER exceeds threshold.
    """
    MyKeyList_A = []
    MyKeyList_B_noisy = []
    MyKeyList_B_corrected = []
    MyKeyList_A_final = []
    MyKeyList_B_final = []
    MyKeyRateList = []
    MyCascadeStats = []

    rng = Random(seed)

    for _ in range(runtimes):
        firstKey, secondKey, startTime, endTime = _run_bb84_once(
            num_bits,
            fibreLen,
            memNoiseMmodel,
            processorNoiseModel,
            qdelay,
            sourceFreq,
            qSpeed,
            cSpeed,
            fibreNoise,
            "BB84+abort",
        )

        effective_init_loss = loss_init
        effective_len_loss = loss_len if loss_len else lenLoss
        firstKey, secondKey = ManualFibreLossModel(
            key1=firstKey,
            key2=secondKey,
            numNodes=2,
            fibreLen=fibreLen,
            iniLoss=effective_init_loss,
            lenLoss=effective_len_loss,
            algorithmFator=2,
        )
        logger.info(
            "After simulated loss: key_len_A=%s key_len_B=%s (init=%s len=%s)",
            len(firstKey),
            len(secondKey),
            effective_init_loss,
            effective_len_loss,
        )

        noisy_b_key = _apply_bit_flip_noise(secondKey, bit_flip_prob, rng)
        ber_before = _bit_error_rate(firstKey, noisy_b_key)
        logger.info(
            "Noise injected: bit_flip_prob=%s ber_before=%s",
            bit_flip_prob,
            ber_before,
        )

        if ber_before > qber_reject_threshold:
            logger.warning(
                "Abort: QBER=%s exceeds threshold=%s",
                ber_before,
                qber_reject_threshold,
            )
            MyKeyList_A.append(firstKey)
            MyKeyList_B_noisy.append(noisy_b_key)
            MyKeyList_B_corrected.append([])
            MyKeyList_A_final.append([])
            MyKeyList_B_final.append([])
            MyKeyRateList.append(0.0)
            MyCascadeStats.append({"aborted": True, "qber": ber_before})
            continue

        corrected_b_key, stats = cascade_reconcile(
            firstKey,
            noisy_b_key,
            block_size=cascade_block_size,
            rounds=cascade_rounds,
            seed=seed,
            final_single_bit_pass=cascade_final_single_bit_pass,
        )
        ber_after = _bit_error_rate(firstKey, corrected_b_key)
        logger.info(
            "Cascade complete: ber_after=%s corrections=%s parity_checks=%s",
            ber_after,
            stats["corrections"],
            stats["parity_checks"],
        )

        pa_len = int(len(firstKey) * pa_output_frac)
        final_a_key = toeplitz_hash(firstKey, pa_len, seed=pa_seed)
        final_b_key = toeplitz_hash(corrected_b_key, pa_len, seed=pa_seed)
        logger.info("Privacy amplification: output_len=%s", pa_len)

        MyKeyList_A.append(firstKey)
        MyKeyList_B_noisy.append(noisy_b_key)
        MyKeyList_B_corrected.append(corrected_b_key)
        MyKeyList_A_final.append(final_a_key)
        MyKeyList_B_final.append(final_b_key)
        MyCascadeStats.append(stats)

        s = SequenceMatcher(None, final_a_key, final_b_key)
        MyKeyRateList.append(len(final_b_key) * s.ratio() / (endTime - startTime) * 10**9)

    return (
        MyKeyList_A,
        MyKeyList_B_noisy,
        MyKeyList_B_corrected,
        MyKeyList_A_final,
        MyKeyList_B_final,
        MyKeyRateList,
        MyCascadeStats,
    )
