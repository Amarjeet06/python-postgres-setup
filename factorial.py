def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# test the function
num = int(input("Enter a number: "))
print(f"Factorial of {num} is {factorial(num)}")
