# export AdvancedHMC:
#     Hamiltonian,
#     sample,
#     update,
#     refresh,
#     transition
export AdvancedHMC

using Random:
    AbstractRNG,
    GLOBAL_RNG
using AdvancedHMC
using AdvancedHMC:
    Hamiltonian,
    AbstractProposal,
    AbstractIntegrator,
    Adaptation,
    PhasePoint,
    phasepoint,
    update,
    refresh,
    transition
import AdvancedHMC: sample

function update_metric(h::Hamiltonian, q::AbstractVector)
    return update(h, q) # Ensure h.metric has the same dim as q.
end

function make_phasepoint(rng::AbstractRNG, h::Hamiltonian, q::AbstractVector)
    p = rand(rng, h.metric)
    z = phasepoint(h, q, p)
    return z
end

make_phasepoint(h, q) = make_phasepoint(GLOBAL_RNG, h, q)

step_size(i::AbstractIntegrator) = i.ϵ

function sample(rng::AbstractRNG, h::Hamiltonian, τ::AbstractProposal, z::PhasePoint)
    z = refresh(rng, z, h)
    z, stat = transition(rng, τ, h, z)
    return z, stat
end

sample(h, τ, z) = sample(GLOBAL_RNG, h, τ, z)
