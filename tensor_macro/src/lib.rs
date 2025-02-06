extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Expr, ExprArray};

#[proc_macro]
pub fn tensor(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
   let input = parse_macro_input!(input as ExprArray);

   // Flatten the nested arrays into a Vec
   let flattened = flatten_array(&input);

   // Generate the output code
   let output = quote! {
      vec![#(#flattened),*]
   };

   // Convert the output back into a TokenStream
   TokenStream::from(output)
}

// Recursively flatten nested arrays
fn flatten_array(array: &ExprArray) -> Vec<Expr> {
   let mut result = Vec::new();
   for element in &array.elems {
      if let Expr::Array(nested_array) = element {
         result.extend(flatten_array(nested_array));
      } else {
         result.push(element.clone());
      }
   }
   result
}