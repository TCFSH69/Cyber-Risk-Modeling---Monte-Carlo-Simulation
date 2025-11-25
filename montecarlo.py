import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

class AttackRiskSimulator:
    def __init__(self, attacks_data):
        self.attacks = attacks_data
        
    def simulate(self, n_simulations=10000):
        print(f"Satrt simulate Monte Carlo {n_simulations} times...")
        
        total_losses = []
        
        attack_losses = {attack['Attack Name']: [] for attack in self.attacks}
        
        for sim in range(n_simulations):
            sim_total_loss = 0
            
            for attack in self.attacks:
                # Poisson Distribution
                n_attacks = np.random.poisson(attack['Average Attack Per Year'])
                
                attack_loss = 0
                for _ in range(n_attacks):
                    if np.random.random() < attack['Success Rate']:
                        sigma = (math.log(attack['Loss Max']) - math.log(attack['Loss Min'])) / 4.652 # 1% and 99% z-score
                        mu = math.log(attack['Loss Min']) + 2.326 * sigma
                        loss = np.random.lognormal(mu, sigma)
                        attack_loss += loss
                
                attack_losses[attack['Attack Name']].append(attack_loss)
                sim_total_loss += attack_loss
            
            total_losses.append(sim_total_loss)
        
        self.total_losses = np.array(total_losses)
        self.attack_losses = {k: np.array(v) for k, v in attack_losses.items()}
        
        print("Simulation Done\n")
        return self.total_losses
    
    def get_statistics(self):
        stats_dict = {
            'Average Loss': np.mean(self.total_losses),
            'Median Loss': np.median(self.total_losses),
            'Standard Deviation': np.std(self.total_losses),
            'Min Loss': np.min(self.total_losses),
            'Max Loss': np.max(self.total_losses),
            'Var 95%': np.percentile(self.total_losses, 95),
        }
        
        print("=" * 60)
        print("Stats of Overall Risk")
        print("=" * 60)
        for key, value in stats_dict.items():
            print(f"{key:15s}: ${value:,.2f}")
        print("=" * 60)
        print()
        
        print("=" * 60)
        print("Stats of Each Type of Attacks")
        print("=" * 60)
        for attack_name, losses in self.attack_losses.items():
            print(f"\n{attack_name}:")
            print(f"  Average Loss: ${np.mean(losses):,.2f}")
            print(f"  Median: ${np.median(losses):,.2f}")
            print(f"  Var 95%: ${np.percentile(losses, 95):,.2f}")
        print("=" * 60)
        
        return stats_dict
    
    def plot_results(self):
        plt.figure(figsize=(16, 12))
        
        # Set unit to 1 Million dollars 
        total_losses_m = self.total_losses / 1e6
        
        # 1. Total Loss Histogram
        ax1 = plt.subplot(2, 2, 1)
        ax1.hist(total_losses_m, bins=50, range=(0.1, 5), color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(total_losses_m), color='red', linestyle='--', linewidth=2, label=f'Average: ${np.mean(total_losses_m):,.2f}M')
        ax1.axvline(np.percentile(total_losses_m, 95), color='orange', linestyle='--', linewidth=2, label=f'Var 95%: ${np.percentile(total_losses_m, 95):,.2f}M')
        ax1.set_xlabel('Annual Total Loss (Million $)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Annual Total Loss Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. CDF
        ax2 = plt.subplot(2, 2, 2)
        sorted_losses_m = np.sort(total_losses_m)
        cdf = np.arange(1, len(sorted_losses_m) + 1) / len(sorted_losses_m)
        ax2.plot(sorted_losses_m, cdf * 100, linewidth=2, color='darkgreen')
        ax2.axhline(95, color='orange', linestyle='--', alpha=0.7, label='95th Percentile')
        ax2.axvline(np.percentile(total_losses_m, 95), color='orange', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Annual Total Loss (Million $)', fontsize=12)
        ax2.set_ylabel('Accumulative Percentage (%)', fontsize=12)
        ax2.set_title('CDF', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Total Loss of Each Type of Attacks Comparison
        ax3 = plt.subplot(2, 2, 3)
        attack_names = list(self.attack_losses.keys())
        attack_means = [np.mean(self.attack_losses[name]) / 1e6 for name in attack_names]

        # List only top 10 attacks (adjustable)
        sorted_indices = np.argsort(attack_means)[::-1]
        top_n = min(10, len(attack_names))
        top_names = [attack_names[i] for i in sorted_indices[:top_n]]
        top_means = [attack_means[i] for i in sorted_indices[:top_n]]

        colors = plt.cm.Set3(range(len(top_names)))
        bars = ax3.bar(range(len(top_names)), top_means, color=colors, edgecolor='black', linewidth=1.5)
        ax3.set_xticks(range(len(top_names)))
        ax3.set_xticklabels(top_names, rotation=45, ha='right')
        ax3.set_ylabel('Annual Total Loss (Million $)', fontsize=12)
        ax3.set_title(f'Top {top_n} Attacks by Average Loss', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height, f'${height:,.2f}M', ha='center', va='bottom', fontsize=10)
        
        # 4. Pie Chart
        ax4 = plt.subplot(2, 2, 4)
        contributions = [np.mean(self.attack_losses[name]) / 1e6 for name in attack_names]

        # List only top 10 attacks (adjustable)
        sorted_indices = np.argsort(contributions)[::-1]
        top_n_pie = min(10, len(attack_names))

        pie_labels = []
        pie_values = []
        pie_colors = []
        
        for i in range(top_n_pie):
            idx = sorted_indices[i]
            pie_labels.append(attack_names[idx])
            pie_values.append(contributions[idx])
            pie_colors.append(plt.cm.Set3(i))
        
        # List only top 10 attack types, the rest labeld as "Others" (adjustable)
        if len(attack_names) > top_n_pie:
            others_sum = sum(contributions[sorted_indices[i]] for i in range(top_n_pie, len(attack_names)))
            pie_labels.append('Others')
            pie_values.append(others_sum)
            pie_colors.append('lightgray')
        
        # Show labels larger than 2% (adjustable)
        def autopct_format(pct):
            return f'{pct:.1f}%' if pct > 2 else ''
        
        # Create legends
        wedges, texts, autotexts = ax4.pie(pie_values, autopct=autopct_format, colors=pie_colors, startangle=90, pctdistance=0.85)
        ax4.legend(wedges, pie_labels, title="Attack Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
        ax4.set_title(f'Top {top_n_pie} Attacks Contribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('monte_carlo.png', dpi=300, bbox_inches='tight')
        print("\nSaving monte_carlo.png")
        plt.show()

def load_attacks_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    attacks = df.to_dict('records')
    return attacks

if __name__ == "__main__":
    attacks = load_attacks_from_csv('data.csv')

    simulator = AttackRiskSimulator(attacks)
    
    losses = simulator.simulate(n_simulations=10000)
    
    stats = simulator.get_statistics()
    
    simulator.plot_results()