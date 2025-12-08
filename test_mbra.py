#!/usr/bin/env python3
"""
Test script for ModelBasedReflexAgent with memory buffer.
Tests the memory buffer functionality and run_mbra simulation.
"""

import argparse
from agent import ModelBasedReflexAgent, FitnessWeights
from agario_simulation import AgarioSimulation


def test_mbra_simulation():
    """Test running a simulation with ModelBasedReflexAgent."""
    print("=" * 60)
    print("Testing ModelBasedReflexAgent with Memory Buffer")
    print("=" * 60)
    
    # Create fitness weights
    fitness_weights = FitnessWeights(
        food=0.5,
        time_alive=100.0,
        cells_eaten=50.0,
        score=0.75,
        death=500.0
    )
    
    # Create ModelBasedReflexAgent with memory buffer
    agent = ModelBasedReflexAgent(
        run_interval=0.1,
        fitness_weights=fitness_weights,
        move_sensitivity=50.0,
        decay_factor=0.92,
        priority_threshold=0.1,
        max_memory_size=100,
        distance_weight_factor=500.0
    )
    
    print(f"Agent created with memory buffer:")
    print(f"  - Decay factor: {agent.memory_buffer.decay_factor}")
    print(f"  - Priority threshold: {agent.memory_buffer.priority_threshold}")
    print(f"  - Max memory size per type: {agent.memory_buffer.max_size_per_type}")
    print(f"  - Distance weight factor: {agent.memory_buffer.distance_weight_factor}")
    print()
    
    # Create simulation
    sim = AgarioSimulation(
        view_width=900,
        view_height=600,
        bounds=1500,
        food_count=600,
        virus_count=20
    )
    
    print("Starting MBRA simulation...")
    print("  - Duration: 60 seconds")
    print("  - FPS: 60")
    print("  - Headless: False (will show visualization)")
    print()
    
    # Run simulation
    try:
        fitness = sim.run_mbra(
            agent=agent,
            base_fps=60,
            simulation_duration=60.0,
            headless=False
        )
        
        print()
        print("=" * 60)
        print("Simulation Complete!")
        print("=" * 60)
        print(f"Final Fitness: {fitness:.2f}")
        print()
        print("Memory Buffer Statistics:")
        print(f"  - Threats in memory: {len(agent.memory_buffer.threats)}")
        print(f"  - Prey in memory: {len(agent.memory_buffer.prey)}")
        print(f"  - Foods in memory: {len(agent.memory_buffer.foods)}")
        print(f"  - Viruses in memory: {len(agent.memory_buffer.viruses)}")
        print()
        
        return fitness
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        return None
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_memory_buffer():
    """Test memory buffer functionality."""
    print("=" * 60)
    print("Testing Memory Buffer Functionality")
    print("=" * 60)
    
    from agent import MemoryBuffer, MemoryItem
    
    # Create memory buffer
    buffer = MemoryBuffer(
        decay_factor=0.92,
        priority_threshold=0.1,
        max_size_per_type=10,
        distance_weight_factor=500.0
    )
    
    print("Testing memory buffer operations...")
    print()
    
    # Test adding items
    print("1. Adding items to memory buffer:")
    buffer.add_item("threat", (100.0, 200.0), 0)
    buffer.add_item("prey", (150.0, 250.0), 0, area=50.0)
    buffer.add_item("food", (200.0, 300.0), 0)
    buffer.add_item("virus", (250.0, 350.0), 0)
    
    print(f"   - Threats: {len(buffer.threats)}")
    print(f"   - Prey: {len(buffer.prey)}")
    print(f"   - Foods: {len(buffer.foods)}")
    print(f"   - Viruses: {len(buffer.viruses)}")
    print()
    
    # Test priority decay
    print("2. Testing priority decay:")
    initial_priority = buffer.threats[0].priority
    print(f"   - Initial threat priority: {initial_priority:.4f}")
    
    buffer.decay_all(10)
    decayed_priority = buffer.threats[0].priority
    print(f"   - After 10 ticks: {decayed_priority:.4f}")
    print(f"   - Decay factor: {buffer.decay_factor}")
    print(f"   - Expected: {initial_priority * (buffer.decay_factor ** 10):.4f}")
    print()
    
    # Test updating visible objects
    print("3. Testing update with visible objects:")
    buffer.update_with_visible_objects(
        threats=[(100.0, 200.0), (300.0, 400.0)],
        prey=[(150.0, 250.0)],
        foods=[(200.0, 300.0)],
        viruses=[],
        current_tick=20
    )
    print(f"   - Threats after update: {len(buffer.threats)}")
    print(f"   - Updated threat priority: {buffer.threats[0].priority:.4f} (should be reset)")
    print()
    
    # Test merging
    print("4. Testing object merging:")
    merged_threats, merged_prey, merged_foods, merged_viruses = buffer.get_merged_objects(
        my_pos=(0.0, 0.0),
        current_threats=[(500.0, 600.0)],
        current_prey=[],
        current_foods=[],
        current_viruses=[]
    )
    print(f"   - Merged threats: {len(merged_threats)}")
    print(f"   - Merged prey: {len(merged_prey)}")
    print(f"   - Merged foods: {len(merged_foods)}")
    print(f"   - Merged viruses: {len(merged_viruses)}")
    print()
    
    print("Memory buffer test completed successfully!")
    print("=" * 60)
    print()


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test ModelBasedReflexAgent with memory buffer")
    parser.add_argument('-t', '--test', choices=['simulation', 'memory', 'all'], 
                       default='all', help='Which test to run')
    parser.add_argument('-d', '--duration', type=float, default=60.0,
                       help='Simulation duration in seconds')
    parser.add_argument('-f', '--fps', type=int, default=60,
                       help='Frames per second')
    parser.add_argument('--headless', action='store_true',
                       help='Run simulation without visualization')
    args = parser.parse_args()
    
    if args.test in ['memory', 'all']:
        test_memory_buffer()
    
    if args.test in ['simulation', 'all']:
        fitness_weights = FitnessWeights(
            food=0.5,
            time_alive=100.0,
            cells_eaten=50.0,
            score=0.75,
            death=500.0
        )
        
        agent = ModelBasedReflexAgent(
            run_interval=0.1,
            fitness_weights=fitness_weights,
            move_sensitivity=50.0,
            decay_factor=0.92,
            priority_threshold=0.1,
            max_memory_size=100,
            distance_weight_factor=500.0
        )
        
        sim = AgarioSimulation(
            view_width=900,
            view_height=600,
            bounds=1500,
            food_count=600,
            virus_count=20
        )
        
        print("Starting MBRA simulation...")
        print(f"  - Duration: {args.duration} seconds")
        print(f"  - FPS: {args.fps}")
        print(f"  - Headless: {args.headless}")
        print()
        
        try:
            fitness = sim.run_mbra(
                agent=agent,
                base_fps=args.fps,
                simulation_duration=args.duration,
                headless=args.headless
            )
            
            if fitness is not None:
                print()
                print("=" * 60)
                print("Simulation Complete!")
                print("=" * 60)
                print(f"Final Fitness: {fitness:.2f}")
                print()
                print("Memory Buffer Statistics:")
                print(f"  - Threats in memory: {len(agent.memory_buffer.threats)}")
                print(f"  - Prey in memory: {len(agent.memory_buffer.prey)}")
                print(f"  - Foods in memory: {len(agent.memory_buffer.foods)}")
                print(f"  - Viruses in memory: {len(agent.memory_buffer.viruses)}")
                print()
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        except Exception as e:
            print(f"\nError during simulation: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

