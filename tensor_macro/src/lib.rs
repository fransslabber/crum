extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Expr, ExprArray};

#[proc_macro]
pub fn tensor_flatten(input: TokenStream) -> TokenStream {
   // Parse the input tokens into a syntax tree
   let input = parse_macro_input!(input as ExprArray);

   let mut sizes = Vec::new();
   let mut level:usize = 1;
   // Flatten the nested arrays and collect sizes
   let flattened = flatten_array(&input,&mut sizes,&mut level);

   // Generate the output code
   let output = quote! {
      {
         let flattened = vec![#(#flattened),*];
         let sizes = vec![#(#sizes),*];
         (flattened, sizes)
      }
   };

   // Convert the output back into a TokenStream
   TokenStream::from(output)
}

// Recursively flatten nested arrays and collect sizes
fn flatten_array(array: &ExprArray, sizes: &mut Vec<usize>,level: &mut usize) -> Vec<Expr> {
   let mut flattened = Vec::new();   

   // Record the size of the current level
   if sizes.len() < *level {
      sizes.push(array.elems.len());
   }
   *level = *level + 1;
   // Recursively process each element
   for element in &array.elems {
      if let Expr::Array(nested_array) = element {
         let nested_flattened = flatten_array(nested_array,sizes,level);
         flattened.extend(nested_flattened);
      } else {
         flattened.push(element.clone());
      }
   }
   *level = *level - 1;
   flattened
}