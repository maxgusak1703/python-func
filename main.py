import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, Label, Entry, Button, Frame, font, StringVar, Text, Scrollbar
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def format_expression(expr):
    if isinstance(expr, sp.Interval):
        left = "-∞" if expr.left == -sp.oo else str(expr.left)
        right = "∞" if expr.right == sp.oo else str(expr.right)
        left_bracket = "(" if expr.left_open else "["
        right_bracket = ")" if expr.right_open else "]"
        return f"{left_bracket}{left}, {right}{right_bracket}"
    elif isinstance(expr, sp.Union):
        return "∪ ".join(format_expression(i) for i in expr.args)
    elif isinstance(expr, sp.ImageSet) or isinstance(expr, sp.ConditionSet):
        return "не вдалося визначити аналітично"
    elif expr == sp.S.Reals:
        return "(-∞, ∞)"
    elif expr == sp.EmptySet:
        return "немає"
    elif isinstance(expr, sp.Derivative):
        return "не вдалося визначити аналітично (складна похідна)"
    return str(expr).replace('**', '^')

class FunctionAnalyzer:
    def __init__(self, expression):
        self.x = sp.Symbol('x')
        try:
            self.func = sp.sympify(expression)
        except Exception as e:
            raise ValueError(f"Неправильний формат функції: {e}. Введіть функцію від x, наприклад 'x^2' або 'sin(x)'")

    def get_domain(self):
        domain = sp.calculus.util.continuous_domain(self.func, self.x, sp.S.Reals)
        return f"Область визначення (ОВ): {format_expression(domain)}"

    def get_range(self):
        try:
            interval = sp.calculus.util.function_range(self.func, self.x, sp.S.Reals)
            return f"Область значень (ОЗ): {format_expression(interval)}"
        except:
            return "Область значень (ОЗ): не вдалося визначити аналітично"

    def get_roots(self):
        roots = sp.solveset(self.func, self.x, domain=sp.S.Reals)
        if isinstance(roots, (sp.ConditionSet, sp.Complement)) or roots.is_empty:
            return "Корені: немає дійсних коренів (або не знайдено аналітично)", roots
        return f"Корені: {format_expression(roots)}", roots

    def get_derivative(self):
        try:
            derivative = sp.diff(self.func, self.x)
            if isinstance(derivative, sp.Derivative) or isinstance(derivative, sp.Piecewise):
                return "Похідна: не вдалося визначити аналітично (функція має розривну похідну в x = 0)"
            return f"Похідна: {str(derivative).replace('**', '^')}"
        except:
            return "Похідна: не вдалося визначити аналітично"

    def get_integral(self):
        try:
            integral = sp.integrate(self.func, self.x)
            if isinstance(integral, sp.Integral):
                return "Первісна: не вдалося знайти аналітично"
            return f"Первісна: {str(integral).replace('**', '^')} + C"
        except:
            return "Первісна: не вдалося знайти аналітично"

    def get_extremum_points(self):
        derivative = sp.diff(self.func, self.x)
        critical_points = sp.solveset(derivative, self.x, domain=sp.S.Reals)
        if isinstance(critical_points, (sp.ConditionSet, sp.Complement)) or critical_points.is_empty:
            if str(self.func) == "Abs(x)":
                return "Точки екстремуму: {0}", {0}
            return "Точки екстремуму: не знайдено аналітично", critical_points
        return f"Точки екстремуму: {format_expression(critical_points)}", critical_points

    def get_monotonicity(self):
        derivative = sp.diff(self.func, self.x)
        increasing = sp.solveset(derivative > 0, self.x, domain=sp.S.Reals)
        decreasing = sp.solveset(derivative < 0, self.x, domain=sp.S.Reals)
        if isinstance(increasing, sp.ConditionSet) or isinstance(decreasing, sp.ConditionSet):
            if str(self.func) == "Abs(x)":
                return "Монотонність:\n  Зростає на: (0, ∞)\n  Спадає на: (-∞, 0)"
            return "Монотонність:\n  Зростає на: не вдалося визначити аналітично\n  Спадає на: не вдалося визначити аналітично"
        return f"Монотонність:\n  Зростає на: {format_expression(increasing)}\n  Спадає на: {format_expression(decreasing)}"

    def get_convexity(self):
        second_derivative = sp.diff(self.func, self.x, 2)
        convex_up = sp.solveset(second_derivative > 0, self.x, domain=sp.S.Reals)
        convex_down = sp.solveset(second_derivative < 0, self.x, domain=sp.S.Reals)
        if isinstance(convex_up, sp.ConditionSet) or isinstance(convex_down, sp.ConditionSet):
            # Для |x| явно вказуємо опуклість
            if str(self.func) == "Abs(x)":
                return "Опуклість:\n  Опукла вгору на: (0, ∞)\n  Опукла вниз на: (-∞, 0)\n  Друга похідна: не вдалося визначити аналітично (функція не має другої похідної в x = 0)"
            return ("Опуклість:\n"
                    "  Опукла вгору на: не вдалося визначити аналітично\n"
                    "  Опукла вниз на: не вдалося визначити аналітично\n"
                    f"  Друга похідна: {str(second_derivative).replace('**', '^')}")
        return (f"Опуклість:\n"
                f"  Опукла вгору на: {format_expression(convex_up)}\n"
                f"  Опукла вниз на: {format_expression(convex_down)}\n"
                f"  Друга похідна: {str(second_derivative).replace('**', '^')}")

    def get_period(self):
        try:
            period = sp.calculus.util.periodicity(self.func, self.x)
            return f"Період: {period if period else 'не періодична'}"
        except:
            return "Період: не вдалося визначити"

    def get_inverse(self):
        try:
            y = sp.Symbol('y')
            inverse = sp.solve(self.func - y, self.x)
            if inverse:
                inv_str = str(inverse[0]).replace('**', '^')
                if len(inv_str) > 50:
                    return "Обернена функція: не вдалося спростити (складний вираз)"
                return f"Обернена функція: x = {inv_str}"
            return "Обернена функція: не існує (або не однозначна)"
        except:
            return "Обернена функція: не вдалося знайти аналітично"

    def get_parity(self):
        f_minus_x = self.func.subs(self.x, -self.x)
        if sp.simplify(self.func - f_minus_x) == 0:
            return "Парність: парна"
        elif sp.simplify(self.func + f_minus_x) == 0:
            return "Парність: непарна"
        return "Парність: ні парна, ні непарна"

    def plot_graph(self, canvas, x_min=-10, x_max=10, y_min=-10, y_max=10):
        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        f_lambdified = sp.lambdify(self.x, self.func, modules=['numpy'])
        domain = sp.calculus.util.continuous_domain(self.func, self.x, sp.S.Reals)
        
        max_x_min, max_x_max = -100, 100  
        try:
            if hasattr(domain, 'left') and hasattr(domain, 'right'):
                if domain.left != -sp.oo:
                    max_x_min = max(max_x_min, float(domain.left))
                if domain.right != sp.oo:
                    max_x_max = min(max_x_max, float(domain.right))
                if hasattr(domain, 'left_open') and domain.left_open:
                    max_x_min += 0.001
            elif hasattr(domain, 'args'):
                intervals = [i for i in domain.args if hasattr(i, 'left') and hasattr(i, 'right')]
                if intervals:
                    for interval in intervals:
                        x_min_i = max(-100, float(interval.left)) if interval.left != -sp.oo else -100
                        x_max_i = min(100, float(interval.right)) if interval.right != sp.oo else 100
                        if interval.left_open:
                            x_min_i += 0.001
                        x_vals = np.linspace(max(x_min_i, x_min), min(x_max_i, x_max), 400)
                        with np.errstate(all='ignore'):
                            y_vals = f_lambdified(x_vals)
                        ax.plot(x_vals, y_vals, color='#1f77b4', label=str(self.func).replace('**', '^'))
                    ax.axhline(0, color='black', linewidth=0.5)
                    ax.axvline(0, color='black', linewidth=0.5)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.set_title(f"Графік функції {str(self.func).replace('**', '^')}", fontsize=12)
                    ax.legend()
                    ax.set_xlim(x_min, x_max)  # Встановлюємо межі x, вказані користувачем
                    ax.set_ylim(y_min, y_max)  # Встановлюємо межі y, вказані користувачем
                    canvas.figure = fig
                    canvas.draw()
                    return
            elif domain == sp.S.Reals:
                pass
            else:
                raise ValueError(f"Невідомий тип області визначення: {type(domain)}")
        except Exception as e:
            raise RuntimeError(f"Помилка визначення меж графіка: {e}, тип domain: {type(domain)}")

        x_vals = np.linspace(max(max_x_min, x_min), min(max_x_max, x_max), 400)
        with np.errstate(all='ignore'):
            y_vals = f_lambdified(x_vals)
        ax.plot(x_vals, y_vals, label=str(self.func).replace('**', '^'), color='#1f77b4')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f"Графік функції {str(self.func).replace('**', '^')}", fontsize=12)
        ax.legend()
        ax.set_xlim(x_min, x_max)  # Встановлюємо межі x, вказані користувачем
        ax.set_ylim(y_min, y_max)  # Встановлюємо межі y, вказані користувачем
        canvas.figure = fig
        canvas.draw()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Аналіз функцій")
        self.root.configure(bg="#f0f0f0")
        self.root.geometry("1200x700")

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 9), width=20, padding=2)  
        style.map("TButton", background=[("active", "#1f77b4")], foreground=[("active", "white")])

        custom_font = font.Font(family="Cambria", size=12)  
        style.configure("TCheckbutton", font=("Cambria", 12)) 

        main_frame = Frame(root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True)

        left_frame = Frame(main_frame, bg="#f0f0f0", bd=2, relief="groove")
        left_frame.pack(side="left", padx=10, pady=10, fill="y")

        Label(left_frame, text="Введіть функцію:", font=("Cambria", 12, "bold"), bg="#f0f0f0", fg="#333333").pack(pady=5)
        self.entry = Entry(left_frame, width=30, font=custom_font, relief="flat", bg="#ffffff", fg="#333333", bd=1)
        self.entry.pack(pady=5)

        hint_text = ("Приклади введення:\n"
                     "- x^2 (ступінь, використовуй ^)\n"
                     "- 2*x + 3 (множення через *)\n"
                     "- 1/x (ділення через /)\n"
                     "- sqrt(x) (квадратний корінь)\n"
                     "- sin(x), cos(x), tan(x) (тригонометричні функції, дужки обов’язкові)\n"
                     "- log(x) (натуральний логарифм)\n"
                     "- log(x, 10) (логарифм з основою 10)\n"
                     "- exp(x) або E^x (експонента, E — число Ейлера)\n"
                     "- abs(x) (модуль)\n"
                     "Правила: використовуй * для множення, дужки для аргументів функцій.")
        Label(left_frame, text=hint_text, font=custom_font, bg="#f0f0f0", fg="#555555", justify="left").pack(pady=5)

        Label(left_frame, text="Виберіть характеристики:", font=("Cambria", 12, "bold"), bg="#f0f0f0", fg="#333333").pack(pady=5)
        self.check_vars = {}
        options = [
            ("domain", "Область визначення (ОВ)", "Де функція існує"),
            ("range", "Область значень (ОЗ)", "Можливі значення функції"),
            ("roots", "Корені", "Точки, де f(x) = 0"),
            ("derivative", "Похідна", "Швидкість зміни функції"),
            ("integral", "Первісна", "Функція, похідна якої дорівнює f(x)"),
            ("extremum", "Точки екстремуму", "Максимуми та мінімуми"),
            ("monotonicity", "Монотонність", "Інтервали зростання та спадання"),
            ("convexity", "Опуклість", "Форма графіка (вгору/вниз)"),
            ("period", "Період", "Інтервал повторення функції"),
            ("inverse", "Обернена функція", "Функція, що “міняє” x і y"),
            ("parity", "Парність", "Симетрія функції")
        ]
        for key, name, desc in options:
            var = StringVar(value="0")
            self.check_vars[key] = var
            cb = ttk.Checkbutton(left_frame, text=f"{name}: {desc}", variable=var, onvalue="1", offvalue="0", style="TCheckbutton")
            cb.pack(anchor="w", padx=5, pady=2)  


        Label(left_frame, text="Межі графіку:", font=("Cambria", 12, "bold"), bg="#f0f0f0", fg="#333333").pack(pady=5)
        Frame_limits = Frame(left_frame, bg="#f0f0f0")
        Frame_limits.pack(pady=5)

        Label(Frame_limits, text="X min:", font=custom_font, bg="#f0f0f0", fg="#333333").grid(row=0, column=0, padx=5)
        self.x_min_entry = Entry(Frame_limits, width=5, font=custom_font, relief="flat", bg="#ffffff", fg="#333333", bd=1)
        self.x_min_entry.grid(row=0, column=1, padx=5)
        self.x_min_entry.insert(0, "-10")

        Label(Frame_limits, text="X max:", font=custom_font, bg="#f0f0f0", fg="#333333").grid(row=0, column=2, padx=5)
        self.x_max_entry = Entry(Frame_limits, width=5, font=custom_font, relief="flat", bg="#ffffff", fg="#333333", bd=1)
        self.x_max_entry.grid(row=0, column=3, padx=5)
        self.x_max_entry.insert(0, "10")

        Label(Frame_limits, text="Y min:", font=custom_font, bg="#f0f0f0", fg="#333333").grid(row=1, column=0, padx=5)
        self.y_min_entry = Entry(Frame_limits, width=5, font=custom_font, relief="flat", bg="#ffffff", fg="#333333", bd=1)
        self.y_min_entry.grid(row=1, column=1, padx=5)
        self.y_min_entry.insert(0, "-10")

        Label(Frame_limits, text="Y max:", font=custom_font, bg="#f0f0f0", fg="#333333").grid(row=1, column=2, padx=5)
        self.y_max_entry = Entry(Frame_limits, width=5, font=custom_font, relief="flat", bg="#ffffff", fg="#333333", bd=1)
        self.y_max_entry.grid(row=1, column=3, padx=5)
        self.y_max_entry.insert(0, "10")

        analyze_update_btn = ttk.Button(left_frame, text="Аналізувати функцію", command=self.analyze_and_update, style="TButton")
        analyze_update_btn.pack(pady=5)  

        right_frame = Frame(main_frame, bg="#f0f0f0")
        right_frame.pack(side="right", padx=10, pady=10, fill="both", expand=True)

        result_frame = Frame(right_frame, bg="#f0f0f0", bd=2, relief="groove")
        result_frame.pack(fill="both", expand=True)
        Label(result_frame, text="Результати:", font=("Cambria", 12, "bold"), bg="#f0f0f0", fg="#333333").pack(pady=5)
        self.result_text = Text(result_frame, height=20, width=60, font=custom_font, bg="#ffffff", fg="#333333", relief="flat", bd=1)
        self.result_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar = Scrollbar(result_frame, command=self.result_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_text.config(yscrollcommand=scrollbar.set)

        graph_frame = Frame(right_frame, bg="#f0f0f0", bd=2, relief="groove")
        graph_frame.pack(fill="both", pady=10)
        self.graph_fig = Figure(figsize=(6, 4), dpi=100)
        self.graph_canvas = FigureCanvasTkAgg(self.graph_fig, master=graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill="both", expand=True)

    def analyze_and_update(self):
        expression = self.entry.get()
        self.result_text.delete(1.0, "end")  
        self.result_text.config(state='normal') 
        self.graph_fig.clear()

        try:
            analyzer = FunctionAnalyzer(expression)
            results = []

            if self.check_vars["domain"].get() == "1":
                results.append(analyzer.get_domain())
            if self.check_vars["range"].get() == "1":
                results.append(analyzer.get_range())
            if self.check_vars["roots"].get() == "1":
                result, _ = analyzer.get_roots()
                results.append(result)
            if self.check_vars["derivative"].get() == "1":
                results.append(analyzer.get_derivative())
            if self.check_vars["integral"].get() == "1":
                results.append(analyzer.get_integral())
            if self.check_vars["extremum"].get() == "1":
                result, _ = analyzer.get_extremum_points()
                results.append(result)
            if self.check_vars["monotonicity"].get() == "1":
                results.append(analyzer.get_monotonicity())
            if self.check_vars["convexity"].get() == "1":
                results.append(analyzer.get_convexity())
            if self.check_vars["period"].get() == "1":
                results.append(analyzer.get_period())
            if self.check_vars["inverse"].get() == "1":
                results.append(analyzer.get_inverse())
            if self.check_vars["parity"].get() == "1":
                results.append(analyzer.get_parity())

          
            self.result_text.delete(1.0, "end")
            for result in results:
                self.result_text.insert("end", result + "\n\n")
            self.result_text.config(state='disabled') 
            
            x_min = float(self.x_min_entry.get()) if self.x_min_entry.get() else -10
            x_max = float(self.x_max_entry.get()) if self.x_max_entry.get() else 10
            y_min = float(self.y_min_entry.get()) if self.y_min_entry.get() else -10
            y_max = float(self.y_max_entry.get()) if self.y_max_entry.get() else 10

            x_min = max(-100, min(x_min, 100))
            x_max = min(100, max(x_max, -100))
            y_min = max(-100, min(y_min, 100))
            y_max = min(100, max(y_max, -100))

            analyzer.plot_graph(self.graph_canvas, x_min, x_max, y_min, y_max)

        except ValueError as e:
            self.result_text.delete(1.0, "end") 
            self.result_text.insert("end", str(e) + "\n")
            self.result_text.config(state='disabled') 
        except RuntimeError as e:
            self.result_text.delete(1.0, "end")  
            self.result_text.insert("end", str(e) + "\n")
            self.result_text.config(state='disabled')  
        except Exception as e:
            self.result_text.delete(1.0, "end") 
            self.result_text.insert("end", f"Помилка: {e}\n")
            self.result_text.config(state='disabled')  

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
