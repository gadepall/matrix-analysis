options=['variable','value']
user_input=''
input_message = "Pick an option:\n"
for index, item in enumerate(options):
  input_message+= f'{index+1}) {item}\n'
input_message += 'coefficient type :'
while user_input.lower()not in options:
  user_input = input(input_message)
print('The type of coefficient is '+ user_input)
if(user_input== 'value'):
        print(input("a:")+"x+"+ (input("b:"))+"y+"+ (input("c:")) + "z ="+ (input("d:")))
else:
        print("x="+(input("a:")), ",y="+ (input("b:")), ",z=" + (input("c:")),",is equal to " + (input("d:")))

