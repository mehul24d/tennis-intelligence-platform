"""Standalone script demonstrating the decay mechanism works correctly -- run this
directly to SEE the behavior, not just trust that tests passed."""
import sys
sys.path.insert(0, "src")
from tennis_intel.live.ml_informed_markov import ServeReturnPosterior

print("=" * 70)
print("CHECK 1: Backward compatibility (lambda_decay=1.0 must change NOTHING)")
print("=" * 70)
p_old = ServeReturnPosterior.from_pretrained_prior(0.70, 60.0, 0.35, 60.0)
p_new_default = ServeReturnPosterior.from_pretrained_prior(0.70, 60.0, 0.35, 60.0, lambda_decay=1.0)
print(f"Default construction:        mean_serve = {p_old.mean_serve():.6f}")
print(f"Explicit lambda_decay=1.0:   mean_serve = {p_new_default.mean_serve():.6f}")
print(f"Identical: {p_old.mean_serve() == p_new_default.mean_serve()}\n")

print("=" * 70)
print("CHECK 2: Does decay actually make OLD evidence fade?")
print("=" * 70)
p_nodecay = ServeReturnPosterior.from_pretrained_prior(0.50, 20.0, 0.50, 20.0, lambda_decay=1.0)
p_decay = ServeReturnPosterior.from_pretrained_prior(0.50, 20.0, 0.50, 20.0, lambda_decay=0.97)
print("Simulating: 15 straight LOSSES, then 15 straight WINS...")
for _ in range(15):
    p_nodecay = p_nodecay.update_serve(False)
    p_decay = p_decay.update_serve(False)
for _ in range(15):
    p_nodecay = p_nodecay.update_serve(True)
    p_decay = p_decay.update_serve(True)
print(f"No decay (remembers everything equally): mean_serve = {p_nodecay.mean_serve():.4f}")
print(f"With decay (recent wins matter more):    mean_serve = {p_decay.mean_serve():.4f}")
print(f"-> Decay version is HIGHER because it 'forgot' the early losses more: "
      f"{p_decay.mean_serve() > p_nodecay.mean_serve()}\n")

print("=" * 70)
print("CHECK 3: Does decay prevent the posterior from becoming permanently rigid?")
print("=" * 70)
p_long_match = ServeReturnPosterior.from_pretrained_prior(0.65, 20.0, 0.35, 20.0, lambda_decay=0.97)
for _ in range(300):
    p_long_match = p_long_match.update_serve(True)
print(f"After 300 simulated points (a very long match), effective_points_serve = "
      f"{p_long_match.effective_points_serve:.1f}")
print(f"(Without decay this would be 300 -- an immovable posterior by that point.")
print(f" With decay it plateaus near 1/(1-0.97) = {1/(1-0.97):.1f}, staying responsive.)")