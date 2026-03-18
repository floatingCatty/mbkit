Excutive summary:

The code base is targeted for a general toolkit that gives the user a very direct unified api to construct the quantum many body Hamiltonian (electronic first, and add bossons later). The api need to be very easy to use, and very intuitive to build new Hamiltonian.

The target customer of this package will be the:
1. the quantum computing simulation people that need to simulate the many-body Hamiltonian of a quantum circuit system.
2. the researcher in theoretical condense matter physics.
3. the developer that build the TCAD tools for quantum devices.

The code unified the way to build Hamiltonian, and then, by adding a intermidiate layer, the Hamiltonian operator can be transformed into format that support different solvers, including but not limited with: the ED solver, the DMRG solver, the CCSD solver, the NQS solver etc. We shoud not build solver ourself to avoid reinvent of wheels. We should use the most advanced solution available. The properties will be extracted with unified api also, to ease the mental burden.

For assist efficient simulation, the code should support many different Hamiltonian format transformation to make it suitable for certain solvers.