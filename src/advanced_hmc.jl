export AdvancedHMC

using Random:
    AbstractRNG,
    GLOBAL_RNG
using AdvancedHMC
using AdvancedHMC:
    Hamiltonian,
    AbstractProposal,
    AbstractIntegrator,
    PhasePoint,
    resize,
    sample_init,
    step
import AdvancedHMC: sample

function update_metric(h::Hamiltonian, q::AbstractVector)
    return resize(h, q) # Ensure h.metric has the same dim as q.
end

function make_phasepoint(rng::AbstractRNG, h::Hamiltonian, q::AbstractVector)
    h, t = sample_init(rng, h, q)
    return t.z
end

make_phasepoint(h, q) = make_phasepoint(GLOBAL_RNG, h, q)

position(z::PhasePoint) = z.θ

momentum(z::PhasePoint) = z.r

step_size(i::AbstractIntegrator) = i.ϵ

function sample(rng::AbstractRNG, h::Hamiltonian, τ::AbstractProposal, z::PhasePoint)
    t = step(rng, h, τ, z)
    return t.z, t.stat
end

sample(h, τ, z) = sample(GLOBAL_RNG, h, τ, z)
