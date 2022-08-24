using Random

abstract type OptimizationType end
abstract type Minimization <: OptimizationType end
abstract type Maximization <: OptimizationType end

Base.show(io::IO, ::Type{Maximization}) = print(io, "↑")
Base.show(io::IO, ::Type{Minimization}) = print(io, "↓")

fitter(::Type{Maximization}, a, b) = max(a, b)
fitter(::Type{Minimization}, a, b) = min(a, b)
fittest(::Type{Maximization}, v) = maximum(v)
fittest(::Type{Minimization}, v) = minimum(v)

struct WithFitness{F,T,O<:OptimizationType}
    value :: T
    fitness :: Float64
    WithFitness{F,O}(t::T) where {F,T,O} = new{F,T,O}(t, F(t))
    WithFitness(t::T; fitness::Function=fitness, optimization::Type{O}=Minimization) where {T,O<:OptimizationType} = WithFitness{fitness,optimization}(t)
end

value(x::WithFitness) = x.value
fitness(x::WithFitness) = x.fitness
Base.isless(a::WithFitness, b::WithFitness) = isless(fitness(a), fitness(b))
fitter(a::WithFitness{F,T,O}, b::WithFitness{F,T,O}) where {F,T,O} = fitter(O, a, b)
Base.show(io::IO, x::WithFitness{F,T,O}) where {F,T,O} = print(io, "WithFitness{$F,$O}($(value(x)), f=$(fitness(x)))")

struct Particle{F,T,O}
    position :: WithFitness{F,T,O}
    velocity :: T 
    best :: WithFitness{F,T,O}
end

value(p::Particle) = value(p.position)
velocity(p::Particle) = p.velocity
fitness(p::Particle) = fitness(p.position)
Base.isless(a::Particle, b::Particle) = isless(fitness(a), fitness(b))
Base.show(io::IO, p::Particle{F,T,O}) where {F,T,O} = print(io, "Particle{$F,$O}($(value(p.position)), f=$(round(fitness(p.position); digits=3)), b=$(round(fitness(p.best); digits=3)))")

struct ParticleSampler{F,T,O}
    position :: Tuple{Float64,Float64}
    velocity :: Tuple{Float64,Float64}
end

function Random.rand(rng::AbstractRNG, s::Random.SamplerTrivial{ParticleSampler{F,T,O}}) where {F,T,O}
    pmin, pmax = s.self.position
    vmin, vmax = s.self.velocity
    position = WithFitness{F,O}((pmax .- pmin) .* rand(rng, T) .+ pmin)
    velocity = (vmax .- vmin) .* rand(rng, T) .+ vmin
    Particle{F,T,O}(position, velocity, position)
end

fitter(a::Particle{F,T,O}, b::Particle{F,T,O}) where {F,T,O} = fitter(O, a, b)
fittest(v::Vector{Particle{F,T,O}}) where {F,T,O} = fittest(O, v)

const PSOParameters = NamedTuple{(:ω, :c₁, :c₂), Tuple{Float64, Float64, Float64}}

mutable struct PSO{F,T,O} <: AbstractVector{Particle{F,T,O}}
    particles :: Vector{Particle{F,T,O}}
    best :: WithFitness{F,T,O}

    parameters :: PSOParameters
    generation :: Int

    function PSO{T}(
        fitness::Function=fitness; 
        particles::Integer = 1000, 
        optimization::Type{O} = Minimization, 
        ω = 0.789, 
        c₁ = 1.49445, 
        c₂ = 1.49445,
        velocity::Tuple{Float64,Float64}=(-1, 1),
        position::Tuple{Float64,Float64}=(-1, 1)
    ) where {T,O<:OptimizationType}
        particles = rand(ParticleSampler{fitness,T,O}(position, velocity), particles)
        new{fitness,T,optimization}(particles, maximum(particles).position, PSOParameters((ω, c₁, c₂)), 0)
    end
end

fittest(pso::PSO) = pso.best
Base.size(pso::PSO) = size(pso.particles)
Base.getindex(pso::PSO, i) = getindex(pso.particles, i)
parameters(pso::PSO) = pso.parameters

function step(p::Particle{F,T,O}; pso::PSO{F,T,O}) where {F,T,O}
    r₁, r₂ = rand(), rand()
    ω, c₁, c₂ = parameters(pso)
    newvelocity = ω .* velocity(p) .+ (c₁ * r₁) .* (value(p.best) .- value(p)).+ (c₂ * r₂) .* (value(pso.best) .- value(p))
    newposition = WithFitness{F,O}(value(p.position) .+ newvelocity)
    Particle(newposition, newvelocity, fitter(newposition, p.best))
end

function step!(pso::PSO; times=1)
    for i in 1:times
        pso.particles .= step.(pso.particles; pso)
        pso.best = fitter(fittest(pso.particles).position, pso.best)
    end
    pso.generation += times
    pso
end
