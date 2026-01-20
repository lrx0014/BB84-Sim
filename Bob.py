from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumProgram
from netsquid.components.instructions import INSTR_MEASURE,INSTR_MEASURE_X
import netsquid as ns

from utils import Random_basis_gen,CompareBasis

import logging
logger = logging.getLogger(__name__)


'''
To add value -1 on tarList at positions given by lossList.
input:
    lossList: A list indecates positions to add -1. (list of int)
    tarList: List to add -1. (list of any)
output:
    tarList: target List.(list of any)
'''
def AddLossCase(lossList,tarList):
    for i in range(len(lossList)):
        tarList.insert(lossList[i],-1)
    return tarList


'''
input:
    Pg: A quantum program (QuantumProgram)
output:
    resList: A list of outputs from the given quantum program, 
    also sorted by key.(list of int)
'''
class QG_B_measure(QuantumProgram):
    def __init__(self,basisList,num_bits):
        self.basisList=basisList
        self.num_bits=num_bits
        super().__init__()


    def program(self):   
        if len(self.basisList)!=self.num_bits:
            logger.error("B measurement error! List length did not match!")

        else:
            for i in range(len(self.basisList)):
                if self.basisList[i] == 0:                  # standard basis
                    self.apply(INSTR_MEASURE, 
                        qubit_indices=i, output_key=str(i),physical=True) 
                else:                              # 1 case # Hadamard basis
                    self.apply(INSTR_MEASURE_X, 
                        qubit_indices=i, output_key=str(i),physical=True) 
        
        yield self.run(parallel=False)



class Protocol(NodeProtocol):
    
    def __init__(self,node,processor,num_bits,
                port_names=["portQB_1","portCB_1","portCB_2"]):
        super().__init__()
        self.num_bits=num_bits
        self.node=node
        self.processor=processor
        self.qList=None
        self.loc_measRes=[]   
        self.basisList=Random_basis_gen(self.num_bits)
        self.portNameQ1=port_names[0]
        self.portNameC1=port_names[1] #to
        self.portNameC2=port_names[2] #from

        self.key=[]
        self.endTime=None


    def run(self):
        
        qubitList=[]
        
        #receive qubits from A
        
        port = self.node.ports[self.portNameQ1]
        yield self.await_port_input(port)
        qubitList.append(port.rx_input().items)
        #logger.info("B received qubits:%s", qubitList)
        
        #put qubits into B memory
        for qubit in qubitList:
            self.processor.put(qubit)
        
        self.myQG_B_measure=QG_B_measure(basisList=self.basisList,num_bits=self.num_bits)
        self.processor.execute_program(self.myQG_B_measure,qubit_mapping=[i for  i in range(self.num_bits)])
        
        yield self.await_program(processor=self.processor)

        # get meas result
        for i in range(self.num_bits):
            tmp=self.myQG_B_measure.output[str(i)][0]
            self.loc_measRes.append(tmp)
        
        logger.info("B meas res:%s", self.loc_measRes)
        
        # self.B_send_basis()
        self.node.ports[self.portNameC1].tx_output(self.basisList)
        
        # receive A's basisList
        port=self.node.ports[self.portNameC2]
        yield self.await_port_input(port)
        basis_A=port.rx_input().items
        logger.info("B received basis_A:%s", basis_A)
        
        self.key=CompareBasis(self.basisList,basis_A,self.loc_measRes)
        
        logger.info("B key:%s", self.key)

        self.endTime=ns.util.simtools.sim_time(magnitude=ns.NANOSECOND)
